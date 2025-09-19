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


__global__ void scan(device_pointers d_p, unsigned k, unsigned level, unsigned V, unsigned* buffers, unsigned* bufferTails, unsigned* visited, degree* core) {
	__shared__ unsigned* buffer;
	__shared__ unsigned bufferTail;

	if (IS_MAIN_THREAD) {
		bufferTail = 0;
		buffer = buffers + blockIdx.x * BUFFER_SIZE;
	}
	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;

		if (visited[v]) continue;	// actually not needed anymore but lets keep him
		if (v >= V) continue;

		if (d_p.out_degrees[v] == level) {
		// if (d_p.out_degrees[v] == level && atomicTestAndSet(&visited[v])) {
			visited[v] = true;
			unsigned loc = atomicAdd(&bufferTail, 1);
			writeToBuffer(buffer, loc, v);
			core[v] = level;
		}
		else if (d_p.in_degrees[v] < k) {
		// else if (d_p.in_degrees[v] < k && atomicTestAndSet(&visited[v])) {
			visited[v] = true;
			unsigned loc = atomicAdd(&bufferTail, 1);
			writeToBuffer(buffer, loc, v);
			d_p.out_degrees[v] = level;
			core[v] = level;
		}
	}
	__syncthreads();

	if (IS_MAIN_THREAD) {
		bufferTails[blockIdx.x] = bufferTail;
	}
}

__global__ void process(device_pointers d_p, unsigned k, unsigned level, unsigned V, unsigned* buffers, unsigned* bufferTails, unsigned* visited, int* core, unsigned int* global_count) {
	__shared__ unsigned bufferTail;
	__shared__ unsigned* buffer;
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

		unsigned v = readFromBuffer(buffer, i);
		unsigned inStart	= d_p.in_neighbors_offset[v];
		unsigned inEnd		= d_p.in_neighbors_offset[v + 1];
		unsigned outStart	= d_p.out_neighbors_offset[v];
		unsigned outEnd		= d_p.out_neighbors_offset[v + 1];

		while (true) {
			__syncwarp();
			if (outStart >= outEnd) break;

			unsigned j = outStart + LANE_ID;
			outStart += WARP_SIZE;

			if (j < outEnd) {
				vertex u = d_p.out_neighbors[j];
				degree uInDegree = atomicSub(d_p.in_degrees + u, 1);

				if (uInDegree == k && atomicTestAndSet(&visited[u])) {
					unsigned loc = atomicAdd(&bufferTail, 1);
					writeToBuffer(buffer, loc, u);
					core[u] = level;
				}
			}
		}

		while (true) {
			__syncwarp();
			if (inStart >= inEnd) break;

			unsigned j = inStart + LANE_ID;
			inStart += WARP_SIZE;

			if (j < inEnd) {
				vertex w = d_p.in_neighbors[j];
				degree wOutDegree = atomicSub(d_p.out_degrees + w, 1);

				if (wOutDegree == (level + 1) && atomicTestAndSet(&visited[w])) {
					unsigned loc = atomicAdd(&bufferTail, 1);
					writeToBuffer(buffer, loc, w);
					core[w] = level;
				}
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


void dcore(Graph &g) {
	bool debug = false;

	vector<degree*> res;

	// setting up GPU memory
	auto startMemory = chrono::steady_clock::now();

	unsigned* buffers = nullptr;
	unsigned* bufferTails = nullptr;
	unsigned* global_count;
	unsigned* visited = nullptr;
	degree* core = nullptr;

	cudaMalloc(&buffers, BUFFER_SIZE * BLOCK_NUMS * sizeof(unsigned));
	cudaMalloc(&bufferTails, BLOCK_NUMS * sizeof(unsigned));
	cudaMalloc(&global_count, sizeof(unsigned));
	cudaMalloc(&visited, g.V * sizeof(unsigned));
	cudaMalloc(&core, g.V * sizeof(degree));

	// moving to GPU
	device_pointers d_p;
	moveGraphToGPU(g, d_p);

	auto endMemory = chrono::steady_clock::now();
	cout << "D-core memory setup done\t" << chrono::duration_cast<chrono::milliseconds>(endMemory - startMemory).count() << "ms" << endl;


	// kmax!!
	auto startKmax = chrono::steady_clock::now();
	unsigned level = 0;
	unsigned count = 0;
	// do a flip!!
	swap(d_p.in_degrees, d_p.out_degrees);
	swap(d_p.in_neighbors, d_p.out_neighbors);
	swap(d_p.in_neighbors_offset, d_p.out_neighbors_offset);

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
	// lets fix the mess we made...
	swap(d_p.in_degrees, d_p.out_degrees);
	swap(d_p.in_neighbors, d_p.out_neighbors);
	swap(d_p.in_neighbors_offset, d_p.out_neighbors_offset);
	refreshGraphOnGPU(g, d_p);

	auto endKmax = chrono::steady_clock::now();
	cout << "D-core k-max done\t\t" << chrono::duration_cast<chrono::milliseconds>(endKmax - startKmax).count() << "ms" << endl;
	cout << "\tkmax: " << kmax << endl;


	// time for the decomposition
	auto startDecomp = chrono::steady_clock::now();
	degree lmax = 0;
	for (unsigned k = 0; k <= kmax; ++k) {

		refreshGraphOnGPU(g, d_p);
		if (debug) cout << "Refreshed graph on device..." << endl;

		level = 0;
		count = 0;

		cudaMemset(global_count, 0, sizeof(unsigned));
		cudaMemset(visited, 0, g.V * sizeof(unsigned));
		cudaMemset(core, -1, g.V * sizeof(degree));

		if (debug) cout << "D-core Computation Started for k=" << k << endl;

		auto start = chrono::steady_clock::now();

		while (count < g.V) {
			cudaMemset(bufferTails, 0, BLOCK_NUMS * sizeof(unsigned));

			scan<<<BLOCK_NUMS, BLOCK_DIM>>>(d_p, k, level, g.V, buffers, bufferTails, visited, core);
			process<<<BLOCK_NUMS, BLOCK_DIM>>>(d_p, k, level, g.V, buffers, bufferTails, visited, core, global_count);

			cudaMemcpy(&count, global_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
			level++;
		}

		lmax = max(lmax, level - 1);
		if (debug) cout << "l-max " << level-1 << endl;

		auto end = chrono::steady_clock::now();
		if (debug) cout << "D-core done, it took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

		auto r = getResultFromGPU(core, g.V);

		res.push_back(r);
	}

	auto endDecomp = chrono::steady_clock::now();
	cout << "D-core decomp done\t\t" << chrono::duration_cast<chrono::milliseconds>(endDecomp - startDecomp).count() << "ms" << endl;

	cout << "\tlmax: " << lmax << endl;

	long long width = to_string(lmax).length();
	ofstream outfile ("../results/cudares.txt",ios::in|ios::out|ios::binary|ios::trunc);
	for (int i = 0; i < res.size(); i++) {
		for(int j = 0; j < g.V; j++){
			outfile << setw(width);
			outfile << res[i][j] << " ";
		}
		outfile << "\r\n";
	}
}

int main(int argc, char *argv[]) {
	const string filename = "../dataset/wiki-vote.txt";

    cout << "Graph loading started... " << endl;
    Graph g(filename);
    cout << ">" << filename << endl;
    cout << "V: " << g.V << endl;
    cout << "E: " << g.E << endl;

	dcore(g);
}
