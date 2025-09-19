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
__global__ void scan(device_pointers d_p, unsigned k, unsigned level, unsigned V, vertex* buffers, unsigned* bufferTails, unsigned* visited, degree* core) {
	__shared__ vertex* buffer;
	__shared__ unsigned bufferTail;

	if (IS_MAIN_THREAD) {
		bufferTail = 0;
		buffer = buffers + blockIdx.x * BUFFER_SIZE;
	}
	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;

		if (visited[v]) continue;
		if (v >= V) continue;

		if (d_p.out_degrees[v] == level) {
			visited[v] = true;
			unsigned loc = atomicAdd(&bufferTail, 1);
			writeVertexToBuffer(buffer, loc, v);
			core[v] = level;
		}
		else if (d_p.in_degrees[v] < k) {
			visited[v] = true;
			unsigned loc = atomicAdd(&bufferTail, 1);
			writeVertexToBuffer(buffer, loc, v);
			d_p.out_degrees[v] = level;
			core[v] = level;
		}
	}
	__syncthreads();

	if (IS_MAIN_THREAD) {
		bufferTails[blockIdx.x] = bufferTail;
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
__global__ void process(device_pointers d_p, unsigned k, unsigned level, unsigned V, vertex* buffers, unsigned* bufferTails, unsigned* visited, degree* core, unsigned int* global_count) {
	__shared__ unsigned bufferTail;
	__shared__ vertex* buffer;
	__shared__ unsigned base;
	unsigned regTail;
	unsigned i;

	if (IS_MAIN_THREAD) {
		base = 0;

		bufferTail = bufferTails[blockIdx.x];
		buffer = buffers + blockIdx.x * BUFFER_SIZE;
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
		offset inStart	= d_p.in_neighbors_offset[v];
		offset inEnd	= d_p.in_neighbors_offset[v + 1];
		offset outStart	= d_p.out_neighbors_offset[v];
		offset outEnd	= d_p.out_neighbors_offset[v + 1];

		for (offset o = outStart; o < outEnd; o += WARP_SIZE) {
			if (o + LANE_ID >= outEnd) continue;

			vertex u = d_p.out_neighbors[o + LANE_ID];
			degree u_in_degree = atomicSub(d_p.in_degrees + u, 1);

			if (u_in_degree == k && atomicTestAndSet(&visited[u])) {
				unsigned loc = atomicAdd(&bufferTail, 1);
				writeVertexToBuffer(buffer, loc, u);
				core[u] = level;
			}
		}

		for (offset o = inStart; o < inEnd; o += WARP_SIZE) {
			if (o + LANE_ID >= inEnd) continue;

			vertex w = d_p.in_neighbors[o + LANE_ID];
			degree w_out_degree = atomicSub(d_p.out_degrees + w, 1);

			if (w_out_degree == (level + 1) && atomicTestAndSet(&visited[w])) {
				unsigned loc = atomicAdd(&bufferTail, 1);
				writeVertexToBuffer(buffer, loc, w);
				core[w] = level;
			}
		}
	}

	__syncthreads();
	if (IS_MAIN_THREAD && bufferTail > 0) {
		atomicAdd(global_count, bufferTail);
	}
}

void moveGraphToGPU(Graph& g, device_pointers& d_p) {
	cudaMalloc(&(d_p.in_neighbors), g.in_neighbors_offset[g.V] * sizeof(vertex));
	cudaMemcpy(d_p.in_neighbors, g.in_neighbors, g.in_neighbors_offset[g.V] * sizeof(vertex), cudaMemcpyHostToDevice);
	cudaMalloc(&(d_p.out_neighbors), g.out_neighbors_offset[g.V] * sizeof(vertex));
	cudaMemcpy(d_p.out_neighbors, g.out_neighbors, g.out_neighbors_offset[g.V] * sizeof(vertex), cudaMemcpyHostToDevice);
	cudaMalloc(&(d_p.in_neighbors_offset), (g.V+1) * sizeof(offset));
	cudaMemcpy(d_p.in_neighbors_offset, g.in_neighbors_offset, (g.V+1) * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMalloc(&(d_p.out_neighbors_offset), (g.V+1) * sizeof(offset));
	cudaMemcpy(d_p.out_neighbors_offset, g.out_neighbors_offset, (g.V+1) * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMalloc(&(d_p.in_degrees), (g.V) * sizeof(degree));
	cudaMemcpy(d_p.in_degrees, g.in_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMalloc(&(d_p.out_degrees), (g.V) * sizeof(degree));
	cudaMemcpy(d_p.out_degrees, g.out_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
}

void refreshGraphOnGPU(Graph& g, device_pointers& d_p) {
	cudaMemcpy(d_p.in_degrees, g.in_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMemcpy(d_p.out_degrees, g.out_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
}

degree* getResultFromGPU(degree* core, unsigned size) {
	auto res = new degree[size];
	cudaMemcpy(res, core, size * sizeof(degree), cudaMemcpyDeviceToHost);

	return res;
}


void dcore(Graph &g, const string& outputFile) {
// ***** setting up GPU memory *****
	auto startMemory = chrono::steady_clock::now();

	unsigned* buffers = nullptr;
	unsigned* bufferTails = nullptr;
	unsigned* global_count;
	unsigned* visited = nullptr;
	degree* core = nullptr;

	cudaMalloc(&buffers, BUFFER_SIZE * BLOCK_NUMS * sizeof(vertex));
	cudaMalloc(&bufferTails, BLOCK_NUMS * sizeof(unsigned));
	cudaMalloc(&global_count, sizeof(unsigned));
	cudaMalloc(&visited, g.V * sizeof(unsigned));	// unsigned instead of bool since atomicCAS only doesn't do bools
	cudaMalloc(&core, g.V * sizeof(degree));

	// moving to GPU
	device_pointers d_p;
	moveGraphToGPU(g, d_p);

	auto endMemory = chrono::steady_clock::now();
	cout << "D-core memory setup done\t" << chrono::duration_cast<chrono::milliseconds>(endMemory - startMemory).count() << "ms" << endl;


// ***** calculating k-max *****
	auto startKmax = chrono::steady_clock::now();
	unsigned level = 0;
	unsigned count = 0;
	swapInOut(d_p); // do a flip!! (we're calculating the 0 l-list)

	cudaMemset(global_count, 0, sizeof(unsigned));
	cudaMemset(visited, 0, g.V * sizeof(unsigned));
	cudaMemset(core, -1, g.V * sizeof(degree));

	while (count < g.V) {
		cudaMemset(bufferTails, 0, BLOCK_NUMS * sizeof(unsigned));

		scan<<<BLOCK_NUMS, BLOCK_DIM>>>(d_p, 0, level, g.V, buffers, bufferTails, visited, core);
		process<<<BLOCK_NUMS, BLOCK_DIM>>>(d_p, 0, level, g.V, buffers, bufferTails, visited, core, global_count);

		cudaMemcpy(&count, global_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
		level++;
	}

	unsigned kmax = level - 1;
	swapInOut(d_p); // let's fix the mess we made...
	refreshGraphOnGPU(g, d_p);

	auto endKmax = chrono::steady_clock::now();
	cout << "D-core k-max done\t\t" << chrono::duration_cast<chrono::milliseconds>(endKmax - startKmax).count() << "ms" << endl;
	cout << "\tkmax: " << kmax << endl;


// ***** time to do the d-core decomposition *****
	auto startDecomp = chrono::steady_clock::now();
	vector<degree*> res;
	degree lmax = 0;
	for (unsigned k = 0; k <= kmax; ++k) {
		refreshGraphOnGPU(g, d_p);	// degrees will be wrecked from previous calculation

		level = 0;
		count = 0;

		cudaMemset(global_count, 0, sizeof(unsigned));
		cudaMemset(visited, 0, g.V * sizeof(unsigned));
		cudaMemset(core, -1, g.V * sizeof(degree));

		while (count < g.V) {
			cudaMemset(bufferTails, 0, BLOCK_NUMS * sizeof(unsigned));

			scan<<<BLOCK_NUMS, BLOCK_DIM>>>(d_p, k, level, g.V, buffers, bufferTails, visited, core);
			process<<<BLOCK_NUMS, BLOCK_DIM>>>(d_p, k, level, g.V, buffers, bufferTails, visited, core, global_count);

			cudaMemcpy(&count, global_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
			level++;
		}

		lmax = max(lmax, level - 1);
		res.push_back(getResultFromGPU(core, g.V));
	}

	auto endDecomp = chrono::steady_clock::now();
	cout << "D-core decomp done\t\t" << chrono::duration_cast<chrono::milliseconds>(endDecomp - startDecomp).count() << "ms" << endl;

	cout << "\tlmax: " << lmax << endl;

// ***** writing out result to file *****
	auto startWrite = chrono::steady_clock::now();
	ofstream bin;
	bin.open(outputFile, ios::binary | ios::out);
	if (bin) {
		for (const auto r: res)
			bin.write(reinterpret_cast<char*>(r), static_cast<streamsize>(g.V * sizeof(degree)));
	} else {
		cout << outputFile << ": could not open file" << endl;
	}
	auto endWrite = chrono::steady_clock::now();
	cout << "D-core results written\t\t" << chrono::duration_cast<chrono::milliseconds>(endWrite - startWrite).count() << "ms" << endl;

	// long long width = to_string(lmax).length();
	// ofstream outfile ("../results/cudares.txt",ios::out|ios::binary|ios::trunc);
	// for (int i = 0; i < res.size(); i++) {
	// 	for(int j = 0; j < g.V; j++){
	// 		outfile << setw(width);
	// 		outfile << res[i][j] << " ";
	// 	}
	// 	outfile << "\r\n";
	// }
}





int main(int argc, char *argv[]) {
	const string filename = "../dataset/amazon0601";

    Graph g(filename);
    cout << "> " << filename  << " V: " << g.V << " E: " << g.E << endl;

	dcore(g, "../results/cudares");
}
