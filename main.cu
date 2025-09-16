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

__global__ void scan(device_pointers d_p, unsigned k, unsigned level, unsigned V, unsigned* buffers, unsigned* bufferTails, bool* visited) {
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

		if (visited[v]) continue;
		if (v >= V) continue;

		if (d_p.out_degrees[v] == level) {
			visited[v] = true;
			unsigned loc = atomicAdd(&bufferTail, 1);
			writeToBuffer(buffer, loc, v);
		}
		if (d_p.in_degrees[v] < k) {
			visited[v] = true;
			unsigned loc = atomicAdd(&bufferTail, 1);
			writeToBuffer(buffer, loc, v);
			d_p.out_degrees[v] = level;
		}
	}
	__syncthreads();

	if (IS_MAIN_THREAD) {
		bufferTails[blockIdx.x] = bufferTail;
	}
}

__global__ void process(device_pointers d_p, unsigned k, unsigned level, unsigned V, unsigned* buffers, unsigned* bufferTails, bool* visited, unsigned int* global_count) {
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

			bool doIn	= inStart < inEnd;
			bool doOut	= outStart < outEnd;

			if (!doIn && !doOut) break;

			if (doOut) {
				unsigned j = outStart + LANE_ID;
				outStart += WARP_SIZE;

				if (j < outEnd) {
					unsigned u = d_p.out_neighbors[j];

					if (!visited[u]) {
						unsigned a = atomicSub(d_p.in_degrees + u, 1);

						if (a <= k) {
							unsigned loc = atomicAdd(&bufferTail, 1);
							writeToBuffer(buffer, loc, u);

							*(d_p.out_degrees + u) = level;

							visited[u] = true;
						}
					}
				}
			}
			if (doIn) {
				unsigned j = inStart + LANE_ID;
				inStart += WARP_SIZE;

				if (j < inEnd) {
					unsigned w = d_p.in_neighbors[j];

					if (!visited[w]) {
						if (*(d_p.out_degrees + w) > level) {
							unsigned a = atomicSub(d_p.out_degrees + w, 1);

							if (a == level + 1) {
								unsigned loc = atomicAdd(&bufferTail, 1);
								writeToBuffer(buffer, loc, w);
								visited[w] = true;
							}
							else if (a <= level) {
								// oops we decremented too much
								atomicAdd(d_p.out_degrees + w, 1);
								visited[w] = true;
							}
						}
					}
				}
			}
		}
	}

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

degree* getResultFromGPU(device_pointers& d_p, unsigned size) {
	auto res = new degree[size];
	cudaMemcpy(res, d_p.out_degrees, size * sizeof(degree), cudaMemcpyDeviceToHost);

	return res;
}

void dcore(Graph &g) {
	unsigned kmax = 15;
	vector<degree*> res;

	bool debug = false;

	auto startDecomp = chrono::steady_clock::now();

	for (unsigned k = 0; k <= kmax; ++k) {
		// moving to GPU
		device_pointers d_p;
		moveGraphToGPU(g, d_p);
		if (debug) cout << "Moved to device..." << endl;

		unsigned level = 0;
		unsigned count = 0;
		unsigned* global_count;
		unsigned* buffers = nullptr;
		unsigned* bufferTails = nullptr;
		bool* visited = nullptr;

		cudaMalloc(&buffers, BUFFER_SIZE * BLOCK_NUMS * sizeof(unsigned));
		cudaMalloc(&bufferTails, BLOCK_NUMS * sizeof(unsigned));
		cudaMalloc(&global_count, sizeof(unsigned));
		cudaMemset(global_count, 0, sizeof(unsigned));
		cudaMalloc(&visited, g.V * sizeof(bool));
		cudaMemset(visited, 0, g.V * sizeof(bool));

		if (debug) cout << "D-core Computation Started for k=" << k << endl;

		auto start = chrono::steady_clock::now();

		while (count < g.V) {
			cudaMemset(bufferTails, 0, BLOCK_NUMS * sizeof(unsigned));

			scan<<<BLOCK_NUMS, BLOCK_DIM>>>(d_p, k, level, g.V, buffers, bufferTails, visited);
			process<<<BLOCK_NUMS, BLOCK_DIM>>>(d_p, k, level, g.V, buffers, bufferTails, visited, global_count);

			cudaMemcpy(&count, global_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
			// cout << "*********Completed level: " << level << ", global_count: " << count << " *********" << endl;
			level++;
		}
		if (debug) cout << "k-max " << level-1 << endl;

		auto end = chrono::steady_clock::now();
		if (debug) cout << "D-core done, it took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

		auto r = getResultFromGPU(d_p, g.V);

		res.push_back(r);
	}

	auto endDecomp = chrono::steady_clock::now();
	cout << "D-core decomp done, it took " << chrono::duration_cast<chrono::milliseconds>(endDecomp - startDecomp).count() << "ms" << endl;

	// unsigned errors = 0;
	// for(int j = 0; j < g.V; j++){
	// 	unsigned degree = res[0][j];
	// 	for (int i = 0; i < res.size(); i++) {
	// 		if (degree != res[i][j])
	// 			errors++;
	// 	}
	// }
	// cout << "errors: " << errors << endl;

	ofstream outfile ("./cudares.txt",ios::in|ios::out|ios::binary|ios::trunc);
	for (int i = 0; i < res.size(); i++) {
		for(int j = 0; j < g.V; j++){
			outfile << setw(3);
			if (res[i][j] > 20) //this should be lmax?
				outfile << -1 << " ";
			else
				outfile << res[i][j] << " ";
		}
		outfile << "\r\n";
	}
}

int main(int argc, char *argv[]) {
	const string filename = "../dataset/congress.txt";

    cout << "Graph loading started... " << endl;
    Graph g(filename);
    cout << ">" << filename << endl;
    cout << "V: " << g.V << endl;
    cout << "E: " << g.E << endl;

	dcore(g);
}
