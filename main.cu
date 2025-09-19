#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ostream>
#include <ranges>
#include <vector>
#include "common.h"
#include "common.cuh"
#include "graph.h"

using namespace std;


/*
 *	Each block gets a buffer
 *	Each thread gets a set of vertices only it accesses, which they then add to their respective block buffer
 *	We do not have to think about atomicity other than for the buffer tails, as we are guaranteed that our set of vertices are unique
 */
__global__ void scan(device_graph_pointers g_p, device_accessory_pointers a_p, unsigned k, unsigned level, unsigned V) {
	__shared__ vertex* buffer;
	__shared__ unsigned bufferTail;

	if (IS_MAIN_THREAD) {
		bufferTail = 0;
		buffer = a_p.buffers + blockIdx.x * BUFFER_SIZE;
	}
	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;

		if (a_p.visited[v]) continue;
		if (v >= V) continue;

		if (g_p.out_degrees[v] == level) {
			a_p.visited[v] = true;
			unsigned loc = atomicAdd(&bufferTail, 1);
			writeVertexToBuffer(buffer, loc, v);
			a_p.core[v] = level;
		}
		else if (g_p.in_degrees[v] < k) {
			a_p.visited[v] = true;
			unsigned loc = atomicAdd(&bufferTail, 1);
			writeVertexToBuffer(buffer, loc, v);
			g_p.out_degrees[v] = level;
			a_p.core[v] = level;
		}
	}
	__syncthreads();

	if (IS_MAIN_THREAD) {
		a_p.bufferTails[blockIdx.x] = bufferTail;
	}
}

/*
 *	Each block have a buffer, which is the same as in the scan phase, we also get the buffer tail from the scan phase
 *	Each warp will process the in- and out-neighbors from a vertex in the block buffer
 *	This will go on, until there are no vertices left in the block buffers
 *	Here we care about atomicity (especially around the visited array, as we should only visit each vertex once)
 *
 *	This kind of wrecks the out degrees in the graph, but it doesn't matter as we ensure that the core array reflects the true out degrees
 */
__global__ void process(device_graph_pointers g_p, device_accessory_pointers a_p, unsigned k, unsigned level) {
	__shared__ unsigned bufferTail;
	__shared__ vertex* buffer;
	__shared__ unsigned base;
	unsigned regTail;
	unsigned i;

	if (IS_MAIN_THREAD) {
		base = 0;

		bufferTail = a_p.bufferTails[blockIdx.x];
		buffer = a_p.buffers + blockIdx.x * BUFFER_SIZE;
		assert(buffer != nullptr);
	}

	__syncthreads();

	while (true) {
		__syncthreads();
		if (base == bufferTail) break;	// every thread exit at the same iteration
		i = base + WARP_ID;
		regTail = bufferTail;
		__syncthreads();

		if (i >= regTail) continue; // this warp won't have to do anything

		if (IS_MAIN_THREAD) {
			base += WARPS_EACH_BLOCK;
			if (regTail < base)
				base = regTail;
		}

		vertex v = readVertexFromBuffer(buffer, i);
		offset inStart	= g_p.in_neighbors_offset[v];
		offset inEnd	= g_p.in_neighbors_offset[v + 1];
		offset outStart	= g_p.out_neighbors_offset[v];
		offset outEnd	= g_p.out_neighbors_offset[v + 1];

		for (offset o = outStart; o < outEnd; o += WARP_SIZE) {
			if (o + LANE_ID >= outEnd) continue;

			vertex u = g_p.out_neighbors[o + LANE_ID];
			degree u_in_degree = atomicSub(g_p.in_degrees + u, 1);

			if (u_in_degree == k && atomicTestAndSet(&a_p.visited[u])) {
				unsigned loc = atomicAdd(&bufferTail, 1);
				writeVertexToBuffer(buffer, loc, u);
				a_p.core[u] = level;
			}
		}

		for (offset o = inStart; o < inEnd; o += WARP_SIZE) {
			if (o + LANE_ID >= inEnd) continue;

			vertex w = g_p.in_neighbors[o + LANE_ID];
			degree w_out_degree = atomicSub(g_p.out_degrees + w, 1);

			if (w_out_degree == (level + 1) && atomicTestAndSet(&a_p.visited[w])) {
				unsigned loc = atomicAdd(&bufferTail, 1);
				writeVertexToBuffer(buffer, loc, w);
				a_p.core[w] = level;
			}
		}
	}

	__syncthreads();
	if (IS_MAIN_THREAD && bufferTail > 0) {
		atomicAdd(a_p.global_count, bufferTail);
	}
}

void allocateDeviceAccesoryMemory(Graph& g, device_accessory_pointers& a_p) {
	cudaMalloc(&(a_p.buffers), BUFFER_SIZE * BLOCK_NUMS * sizeof(vertex));
	cudaMalloc(&(a_p.bufferTails), BLOCK_NUMS * sizeof(unsigned));
	cudaMalloc(&(a_p.global_count), sizeof(unsigned));
	cudaMalloc(&(a_p.visited), g.V * sizeof(unsigned));	// unsigned instead of bool since atomicCAS only doesn't do bools
	cudaMalloc(&(a_p.core), g.V * sizeof(degree));
}

void allocateDeviceGraphMemory(Graph& g, device_graph_pointers& g_p) {
	cudaMalloc(&(g_p.in_neighbors), g.in_neighbors_offset[g.V] * sizeof(vertex));
	cudaMalloc(&(g_p.out_neighbors), g.out_neighbors_offset[g.V] * sizeof(vertex));
	cudaMalloc(&(g_p.in_neighbors_offset), (g.V+1) * sizeof(offset));
	cudaMalloc(&(g_p.out_neighbors_offset), (g.V+1) * sizeof(offset));
	cudaMalloc(&(g_p.in_degrees), (g.V) * sizeof(degree));
	cudaMalloc(&(g_p.out_degrees), (g.V) * sizeof(degree));
}

void moveGraphToDevice(Graph& g, device_graph_pointers& g_p) {
	cudaMemcpy(g_p.in_neighbors, g.in_neighbors, g.in_neighbors_offset[g.V] * sizeof(vertex), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.out_neighbors, g.out_neighbors, g.out_neighbors_offset[g.V] * sizeof(vertex), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.in_neighbors_offset, g.in_neighbors_offset, (g.V+1) * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.out_neighbors_offset, g.out_neighbors_offset, (g.V+1) * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.in_degrees, g.in_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.out_degrees, g.out_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
}

void refreshGraphOnGPU(Graph& g, device_graph_pointers& g_p) {
	cudaMemcpy(g_p.in_degrees, g.in_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.out_degrees, g.out_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
}

pair<degree, vector<degree>> KList(Graph& g, device_graph_pointers& g_p, device_accessory_pointers& a_p, unsigned k) {
	degree lmax = 0;

	unsigned level = 0;
	unsigned count = 0;

	// resetting GPU variables
	cudaMemset(a_p.global_count, 0, sizeof(unsigned));
	cudaMemset(a_p.visited, 0, g.V * sizeof(unsigned));
	cudaMemset(a_p.core, -1, g.V * sizeof(degree));

	// algo time
	while (count < g.V) {
		cudaMemset(a_p.bufferTails, 0, BLOCK_NUMS * sizeof(unsigned));

		scan<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, a_p, k, level, g.V);
		process<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, a_p, k, level);

		cudaMemcpy(&count, a_p.global_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
		level++;
	}

	// getting result from GPU
	vector<degree> res(g.V);
	cudaMemcpy(res.data(), a_p.core, g.V * sizeof(degree), cudaMemcpyDeviceToHost);

	return {max(lmax, level - 1), res};
}



vector<vector<degree>> dcore(Graph &g) {
// ***** setting up GPU memory *****
	auto startMemory = chrono::steady_clock::now();

	// alloc all the memory we need
	device_accessory_pointers a_p;
	allocateDeviceAccesoryMemory(g, a_p);
	device_graph_pointers g_p;
	allocateDeviceGraphMemory(g, g_p);

	// move graph to GPU
	moveGraphToDevice(g, g_p);

	auto endMemory = chrono::steady_clock::now();
	cout << "D-core memory setup done\t" << chrono::duration_cast<chrono::milliseconds>(endMemory - startMemory).count() << "ms" << endl;

// ***** calculating k-max *****
	auto startKmax = chrono::steady_clock::now();
	swapInOut(g_p); // do a flip!! (we're calculating the 0 l-list)
	auto [kmax, _] = KList(g, g_p, a_p, 0);
	swapInOut(g_p); // let's fix the mess we made...

	auto endKmax = chrono::steady_clock::now();
	cout << "D-core k-max done\t\t" << chrono::duration_cast<chrono::milliseconds>(endKmax - startKmax).count() << "ms" << endl;
	cout << "\tkmax: " << kmax << endl;


// ***** time to do the d-core decomposition *****
	auto startDecomp = chrono::steady_clock::now();
	vector<vector<degree>> res;
	degree lmax = 0;
	for (unsigned k = 0; k <= kmax; ++k) {
		refreshGraphOnGPU(g, g_p);	// degrees will be wrecked from previous calculation

		auto [l, core] = KList(g, g_p, a_p, k);
		lmax = max(lmax, l);
		res.push_back(core);
	}

	auto endDecomp = chrono::steady_clock::now();
	cout << "D-core decomp done\t\t" << chrono::duration_cast<chrono::milliseconds>(endDecomp - startDecomp).count() << "ms" << endl;

	cout << "\tlmax: " << lmax << endl;

	return res;
}

void writeDCoreResults(vector<vector<degree>>& values, const string& outputFile) {
	auto startWrite = chrono::steady_clock::now();
	ofstream bin;
	bin.open(outputFile, ios::binary|ios::out|ios::trunc);
	if (bin) {
		for (const auto r: values)
			bin.write(reinterpret_cast<const char*>(r.data()), static_cast<streamsize>(r.size() * sizeof(degree)));
	} else {
		cout << outputFile << ": could not open file" << endl;
	}
	auto endWrite = chrono::steady_clock::now();
	cout << "D-core results written\t\t" << chrono::duration_cast<chrono::milliseconds>(endWrite - startWrite).count() << "ms" << endl;
}

void writeDCoreResultsText(vector<vector<degree>>& values, const string& outputFile, unsigned lmax) {
	auto startWrite = chrono::steady_clock::now();
	// this is for writing to text files
	long long width = to_string(lmax).length();
	ofstream outfile (outputFile,ios::out|ios::binary|ios::trunc);
	for (auto r: values) {
		for (auto v: r) {
			outfile << setw(width);
			outfile << v << " ";
		}
		outfile << "\r\n";
	}
	auto endWrite = chrono::steady_clock::now();
	cout << "D-core results written\t\t" << chrono::duration_cast<chrono::milliseconds>(endWrite - startWrite).count() << "ms" << endl;
}


int main(int argc, char *argv[]) {
	const string filename = "../dataset/wiki_vote";

	auto start = chrono::steady_clock::now();

    Graph g(filename);
    cout << "> " << filename  << " V: " << g.V << " E: " << g.E << endl;

	auto res = dcore(g);
	// writeDCoreResultsText(res, "../results/cudares", 16);
	writeDCoreResults(res, "../results/cudares");

	auto end = chrono::steady_clock::now();
	cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
}
