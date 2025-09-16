#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ostream>
#include <ranges>
#include <vector>

#include "graph.h"

using namespace std;

__device__ inline unsigned readFromBuffer(unsigned* buffer, unsigned loc) {
	assert(loc < BUFFER_SIZE);
	return buffer[loc];
}

__device__ inline void writeToBuffer(unsigned* buffer, unsigned loc, unsigned val) {
	assert(loc < BUFFER_SIZE);
	buffer[loc] = val;
}

__global__ void scan(device_pointers d_p, unsigned k, unsigned level, unsigned V, unsigned* inBuffers, unsigned* inBufferTails, unsigned* outBuffers, unsigned* outBufferTails) {
	__shared__ unsigned* outBuffer;
	__shared__ unsigned outBufferTail;
	__shared__ unsigned* inBuffer;
	__shared__ unsigned inBufferTail;

	if (IS_MAIN_THREAD) {
		outBufferTail = 0;
		outBuffer = outBuffers + blockIdx.x * BUFFER_SIZE;
		inBufferTail = 0;
		inBuffer = inBuffers + blockIdx.x * BUFFER_SIZE;
	}
	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;

		if (v >= V) continue;

		if (d_p.out_degrees[v] == level) {
			unsigned loc = atomicAdd(&outBufferTail, 1);
			writeToBuffer(outBuffer, loc, v);
		}
		else if (d_p.in_degrees[v] < k) {
			unsigned loc = atomicAdd(&inBufferTail, 1);
			writeToBuffer(inBuffer, loc, v);
			d_p.out_degrees[v] = level;
		}
	}
	__syncthreads();

	if (IS_MAIN_THREAD) {
		outBufferTails[blockIdx.x] = outBufferTail;
	}
}

__global__ void process(device_pointers d_p, unsigned level, unsigned V, unsigned* inBuffers, unsigned* inBufferTails, unsigned* outBuffers, unsigned* outBufferTails, unsigned int* global_count) {
	__shared__ unsigned outBufferTail;
	__shared__ unsigned* outBuffer;
	__shared__ unsigned* inBuffer;
	__shared__ unsigned inBufferTail;
	__shared__ unsigned base;
	unsigned regTail;
	unsigned i;

	if (IS_MAIN_THREAD) {
		base = 0;

		outBufferTail = outBufferTails[blockIdx.x];
		outBuffer = outBuffers + blockIdx.x * BUFFER_SIZE;
		assert(outBuffer != nullptr);

		inBufferTail = inBufferTails[blockIdx.x];
		inBuffer = inBuffers + blockIdx.x * BUFFER_SIZE;
		assert(inBuffer != nullptr);
	}

	while (true) {
		__syncthreads();
		if (base == outBufferTail) break;	// every thread exit at the same iteration
		i = base + WARP_ID;
		regTail = outBufferTail;
		__syncthreads();

		if (i >= regTail) continue; // this warp won't have to do anything

		if (IS_MAIN_THREAD) {
			base += WARPS_EACH_BLOCK;
			if (regTail < base)
				base = regTail;
		}

		unsigned v = readFromBuffer(outBuffer, i);
		unsigned start = d_p.out_neighbors_offset[v];
		unsigned end   = d_p.out_neighbors_offset[v + 1];

		while (true) {
			__syncwarp();

			if (start >= end) break;

			unsigned j = start + LANE_ID;
			start += WARP_SIZE;
			if (j >= end) continue;

			unsigned u = d_p.out_neighbors[j];

			if (*(d_p.out_degrees + u) > level) {
				unsigned a = atomicSub(d_p.out_degrees + u, 1);

				if (a == level + 1) {
					unsigned loc = atomicAdd(&outBufferTail, 1);
					writeToBuffer(outBuffer, loc, u);
				}
				else if (a <= level) {
					// oops we decremented too much
					atomicAdd(d_p.out_degrees + u, 1);
				}
			}
		}
	}

	if (IS_MAIN_THREAD && outBufferTail > 0) {
		atomicAdd(global_count, outBufferTail);
	}
}

void dcore(Graph &g) {
	// moving to GPU
	device_pointers d_p;
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
	cout << "Moved to device..." << endl;

	unsigned level = 0;
	unsigned count = 0;
	unsigned* global_count;
	unsigned* inBuffers = nullptr;
	unsigned* inBufferTails = nullptr;
	unsigned* outBuffers = nullptr;
	unsigned* outBufferTails = nullptr;

	cudaMalloc(&inBuffers, BUFFER_SIZE * BLOCK_NUMS * sizeof(unsigned));
	cudaMalloc(&inBufferTails, BLOCK_NUMS * sizeof(unsigned));
	cudaMalloc(&outBuffers, BUFFER_SIZE * BLOCK_NUMS * sizeof(unsigned));
	cudaMalloc(&outBufferTails, BLOCK_NUMS * sizeof(unsigned));
	cudaMalloc(&global_count, sizeof(unsigned));
	cudaMemset(global_count, 0, sizeof(unsigned));

	cout << "D-core Computation Started" << endl;

	auto start = chrono::steady_clock::now();

	while (count < g.V) {
		cudaMemset(inBufferTails, 0, BLOCK_NUMS * sizeof(unsigned));
		cudaMemset(outBufferTails, 0, BLOCK_NUMS * sizeof(unsigned));

		scan<<<BLOCK_NUMS, BLOCK_DIM>>>(d_p, 3, level, g.V, inBuffers, inBufferTails, outBuffers, outBufferTails);
		process<<<BLOCK_NUMS, BLOCK_DIM>>>(d_p, level, g.V, inBuffers, inBufferTails, outBuffers, outBufferTails, global_count);

		cudaMemcpy(&count, global_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
		cout << "*********Completed level: " << level << ", global_count: " << count << " *********" << endl;
		level++;
	}
	cout << "k-max " << level-1 << endl;

	auto end = chrono::steady_clock::now();
	cout << "D-core done, it took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

	cudaMemcpy(g.out_degrees, d_p.out_degrees, (g.V) * sizeof(degree), cudaMemcpyDeviceToHost);

	vector<degree*> res;
	res.push_back(g.out_degrees);

	ofstream outfile ("./cudares.txt",ios::in|ios::out|ios::binary|ios::trunc);
	for (int i = 0; i < res.size(); i++) {
		for(int j = 0; j < g.V; j++){
			outfile << res[i][j] << " ";
		}
		outfile << "\r\n";
	}
}

int main(int argc, char *argv[]) {
	const string filename = "../dataset/digraph.txt";

    cout << "Graph loading started... " << endl;
    Graph g(filename);
    cout << ">" << filename << endl;
    cout << "V: " << g.V << endl;
    cout << "E: " << g.E << endl;

	dcore(g);
}
