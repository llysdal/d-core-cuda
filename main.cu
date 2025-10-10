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
	__shared__ vertex* buffer;
	__shared__ unsigned bufferTail;
	__shared__ unsigned base;
	unsigned vertexIdx;

	if (IS_MAIN_THREAD) {
		base = 0;

		bufferTail = a_p.bufferTails[blockIdx.x];
		buffer = a_p.buffers + blockIdx.x * BUFFER_SIZE;
	}

	__syncthreads();

	while (true) {
		__syncthreads();
		if (base == bufferTail) break;	// no new vertices were added in the last iterations,  we're done
		vertexIdx = base + WARP_ID;
		__syncthreads();

		if (IS_MAIN_THREAD) {
			base += WARPS_EACH_BLOCK;
			if (bufferTail < base)
				// we overshot the buffer tail, let's rewind to make sure we catch the new buffer items
				base = bufferTail;
		}

		if (vertexIdx >= bufferTail) continue;

		vertex v = readVertexFromBuffer(buffer, vertexIdx);
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
			degree w_out_degree = atomicSub(g_p.out_degrees + w, 1);	// this returns previous value

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

void allocateDeviceAccessoryMemory(Graph& g, device_accessory_pointers& a_p) {
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

	return {level - 1, res};
}

vector<vector<degree>> dcore(Graph &g) {
// ***** setting up GPU memory *****
	auto startMemory = chrono::steady_clock::now();

	// alloc all the memory we need
	device_accessory_pointers a_p;
	allocateDeviceAccessoryMemory(g, a_p);
	device_graph_pointers g_p;
	allocateDeviceGraphMemory(g, g_p);

	// move graph to GPU
	moveGraphToDevice(g, g_p);

	auto endMemory = chrono::steady_clock::now();
	cout << "D-core memory setup done\t" << chrono::duration_cast<chrono::milliseconds>(endMemory - startMemory).count() << "ms" << endl;

// ***** calculating k-max *****
	auto startKmax = chrono::steady_clock::now();
	swapInOut(g_p); // do a flip!! (we're calculating the 0 l-list)
	auto [kmax, kmaxes] = KList(g, g_p, a_p, 0);
	swapInOut(g_p); // let's fix the mess we made...

	auto endKmax = chrono::steady_clock::now();
	cout << "D-core k-max done\t\t" << chrono::duration_cast<chrono::milliseconds>(endKmax - startKmax).count() << "ms" << endl;
	cout << "\tkmax: " << kmax << endl;
	g.kmax = kmax;
	g.kmaxes = kmaxes;

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
	g.lmax = lmax;
	g.lmaxes = res;

	cout << "\tlmax: " << lmax << endl;

	return res;
}

vector<degree> checkKmax(Graph &g) {
	// ***** setting up GPU memory *****
	auto startMemory = chrono::steady_clock::now();

	// alloc all the memory we need
	device_accessory_pointers a_p;
	allocateDeviceAccessoryMemory(g, a_p);
	device_graph_pointers g_p;
	allocateDeviceGraphMemory(g, g_p);

	// move graph to GPU
	moveGraphToDevice(g, g_p);

	auto endMemory = chrono::steady_clock::now();
	// cout << "D-core memory setup done\t" << chrono::duration_cast<chrono::milliseconds>(endMemory - startMemory).count() << "ms" << endl;

	// ***** calculating k-max *****
	auto startKmax = chrono::steady_clock::now();
	swapInOut(g_p); // do a flip!! (we're calculating the 0 l-list)
	auto [kmax, kmaxes] = KList(g, g_p, a_p, 0);
	swapInOut(g_p); // let's fix the mess we made...

	auto endKmax = chrono::steady_clock::now();
	// cout << "D-core k-max done\t\t" << chrono::duration_cast<chrono::milliseconds>(endKmax - startKmax).count() << "ms" << endl;
	// cout << "\tkmax: " << kmax << endl;
	// g.kmax = kmax;
	// g.kmaxes = kmaxes;

	return kmaxes;
}

bool compareDCoreResults(const string& first, const string& second) {
	auto startCompare = chrono::steady_clock::now();
	string result = "equal";
	ifstream f1(first, ios::binary|ios::ate);
	ifstream f2(second, ios::binary|ios::ate);

	if (f1.fail() || f2.fail()) {
		result = "file problem";
		auto endCompare = chrono::steady_clock::now();
		cout << "D-core results compared\t\t" << chrono::duration_cast<chrono::milliseconds>(endCompare - startCompare).count() << "ms" << endl;
		cout << "\t" << result << endl;
		return false;
	}
	if (f1.tellg() != f2.tellg()) {
		result = "size mismatch";
		auto endCompare = chrono::steady_clock::now();
		cout << "D-core results compared\t\t" << chrono::duration_cast<chrono::milliseconds>(endCompare - startCompare).count() << "ms" << endl;
		cout << "\t" << result << endl;
		return false;
	}

	f1.seekg(0, ios::beg);
	f2.seekg(0, ios::beg);
	if (!equal(istreambuf_iterator<char>(f1.rdbuf()), istreambuf_iterator<char>(), istreambuf_iterator<char>(f2.rdbuf())))
		result = "not equal";

	auto endCompare = chrono::steady_clock::now();
	cout << "D-core results compared\t\t" << chrono::duration_cast<chrono::milliseconds>(endCompare - startCompare).count() << "ms" << endl;
	cout << "\t" << result << endl;
	return result == "equal";
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

/*
 * we need:
 * kmaxes, original_kmax, compute[bool, n], ED[degree, n], PED[degree, n],
 * tmp_neighbor_in_coreness[degree, n * in degrees]
 * h_index calculator
 */

void allocateDeviceMaintenanceMemory(Graph& g, device_maintenance_pointers& m_p) {
	cudaMalloc(&(m_p.k_max), g.V * sizeof(degree));
	cudaMalloc(&(m_p.original_k_max), g.V * sizeof(degree));
	cudaMalloc(&(m_p.compute), g.V * sizeof(unsigned));
	cudaMalloc(&(m_p.ED), g.V * sizeof(degree));
	cudaMalloc(&(m_p.PED), g.V * sizeof(degree));
	cudaMalloc(&(m_p.tmp_neighbor_in_coreness), g.E * sizeof(unsigned));
	cudaMalloc(&(m_p.hIndex_buckets), g.V + g.E * sizeof(unsigned));
}

void initializeDeviceMaintenanceMemory(Graph& g, device_maintenance_pointers& m_p, vector<degree>& kmax) {
	cudaMemcpy(m_p.k_max, kmax.data(), g.V * sizeof(vertex), cudaMemcpyHostToDevice);
	cudaMemcpy(m_p.original_k_max, kmax.data(), g.V * sizeof(vertex), cudaMemcpyHostToDevice);
	cudaMemset(m_p.compute, 0, g.V * sizeof(unsigned));
	cudaMemset(m_p.tmp_neighbor_in_coreness, 0, g.E * sizeof(unsigned));

	// these shouldnt be needed, for optimization
	cudaMemset(m_p.ED, 0, g.V * sizeof(degree));
	cudaMemset(m_p.PED, 0, g.V * sizeof(degree));
}

__global__ void kmaxCalculateEDandPED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	// ED
	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;

		for (offset o = g_p.in_neighbors_offset[v]; o < g_p.in_neighbors_offset[v + 1]; ++o) {
			vertex neighbor = g_p.in_neighbors[o];
			if (m_p.k_max[neighbor] >= m_p.k_max[v])
				++m_p.ED[v];
		}
	}

	__syncthreads();

	// PED
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		m_p.PED[v] = m_p.ED[v];

		for (offset o = g_p.in_neighbors_offset[v]; o < g_p.in_neighbors_offset[v + 1]; ++o) {
			vertex neighbor = g_p.in_neighbors[o];
			if (m_p.k_max[neighbor] == m_p.k_max[v] && m_p.ED[neighbor] <= m_p.k_max[v])
				--m_p.PED[v];
		}
	}
}

__global__ void kmaxFindUpperBound(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, vertex edgeFrom, vertex edgeTo) {
	vertex root = edgeFrom;
	if (m_p.k_max[edgeTo] < m_p.k_max[edgeFrom])
		root = edgeTo;
	degree root_k_max = m_p.k_max[root];	// should be shared so that we only access memory once

	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;

		if (m_p.k_max[v] == root_k_max && m_p.PED[v] > m_p.k_max[v]) {
			m_p.compute[v] = true;
			++m_p.k_max[v];
		}
	}
}

__device__ unsigned hIndex(device_graph_pointers g_p ,device_maintenance_pointers m_p, vertex v) {
	offset o = g_p.in_neighbors_offset[v];
	degree n = g_p.in_degrees[v];

	unsigned bucketStart = o + v;
	for (int i = 0; i < v + 1; i++)
		m_p.hIndex_buckets[bucketStart + i] = 0;

	for(int i = 0; i < n; i++){
		int x = m_p.tmp_neighbor_in_coreness[o + i];
		if(x >= n){
			m_p.hIndex_buckets[bucketStart + n]++;
		} else {
			m_p.hIndex_buckets[bucketStart + x]++;
		}
	}
	int cnt = 0;
	for (int i = n; i >= 0; i--){
		cnt += m_p.hIndex_buckets[bucketStart + i];
		if (cnt >= i) return i;
	}
	return -1; // should never happen;
}

__global__ void kmaxRefineHIndex(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	__shared__ bool flag;

	if (IS_MAIN_THREAD) flag = true;
	__syncthreads();

	while (flag) {
		if (IS_MAIN_THREAD) flag = false;
		__syncthreads();

		unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
		for (unsigned base = 0; base < V; base += THREAD_COUNT) {
			vertex v = base + global_threadIdx;

			if (v < V && !m_p.compute[v]) {
				for (offset o = g_p.in_neighbors_offset[v]; o < g_p.in_neighbors_offset[v + 1]; ++o) {
					vertex neighbor = g_p.in_neighbors[o];
					if (m_p.k_max[neighbor] == m_p.original_k_max[v]) {
						// add to temp_nieghbor in coreness
						m_p.tmp_neighbor_in_coreness[o] = m_p.k_max[neighbor];
					}
				}
			}

			__syncthreads();

			if (v >= V || m_p.compute[v]) continue;

			// calculate h index
			unsigned tmp_h_index = hIndex(g_p, m_p, v);

			if (tmp_h_index < m_p.k_max[v]) {
				m_p.k_max[v] = tmp_h_index;
				flag = true;
			}
		}
	}
}

void kmaxMaintenance(Graph &g, pair<vertex, vertex> insertedEdge) {
	// cout << "K_max maintenance" << endl;
	// cout << "\tInserting edge " << insertedEdge.first << "->" << insertedEdge.second << endl;
	g.insertEdge(insertedEdge);

	// cout << "\tLoading graph on GPU" << endl;
	device_graph_pointers g_p;
	allocateDeviceGraphMemory(g, g_p);
	// move graph to GPU
	moveGraphToDevice(g, g_p);

	// cout << "\tSetting up maintenance CUDA memory" << endl;
	device_maintenance_pointers m_p;
	allocateDeviceMaintenanceMemory(g, m_p);
	initializeDeviceMaintenanceMemory(g, m_p, g.kmaxes);

	// cout << "\tCalculating ED and PED" << endl;
	kmaxCalculateEDandPED<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V);
	// vector<degree> ED(g.V);
	// vector<degree> PED(g.V);
	// cudaMemcpy(ED.data(), m_p.ED, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	// cudaMemcpy(PED.data(), m_p.PED, g.V * sizeof(degree), cudaMemcpyDeviceToHost);

	// cout << " ED: ";
	// for (auto v: ED)
	// 	cout << v << " ";
	// cout << endl;
	// cout << "PED: ";
	// for (auto v: PED)
	// 	cout << v << " ";
	// cout << endl;

	// cout << "\tCalculating upper bounds" << endl;
	kmaxFindUpperBound<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V, insertedEdge.first, insertedEdge.second);

	// vector<degree> upper(g.V);
	// cudaMemcpy(upper.data(), m_p.k_max, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	// cout << "kmax_upper: ";
	// for (auto v: upper)
	// 	cout << v << " ";
	// cout << endl;
	// vector<unsigned> compute(g.V);
	// cudaMemcpy(compute.data(), m_p.compute, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	// cout << "   compute: ";
	// for (auto v: compute)
	// 	cout << v << " ";
	// cout << endl;

	// cout << "\tRefining with hIndex" << endl;
	kmaxRefineHIndex<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V);

	// vector<unsigned> hindex(g.V + g.E);
	// cudaMemcpy(kmax.data(), m_p.k_max, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	// cout << "kmax_refin: ";
	// for (auto v: kmax)
	// 	cout << v << " ";
	// cout << endl;

	vector<degree> kmax(g.V);
	cudaMemcpy(kmax.data(), m_p.k_max, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "kmax_refin: ";
	for (auto v: kmax)
		cout << v << " ";
	cout << endl;
}

uint32_t cal_hIndex(const vector<uint32_t> &input_vector){
	int n = input_vector.size();
	vector <int> bucket(n + 1);
	for(int i = 0; i < n; i++){
		int x = input_vector[i];
		if(x >= n){
			bucket[n]++;
		} else {
			bucket[x]++;
		}
	}
	int cnt = 0;
	for(int i = n; i >= 0; i--){
		cnt += bucket[i];
		if(cnt >= i)return i;
	}
	return -1;
}

vector<degree> maintainKmax_(Graph& g, vector<pair<vertex, vertex>> modifiedEdges) {
	const unsigned n_ = g.V;
	const int lmax_number_of_threads = 1;

	// typedef struct {
	// 	uint32_t vid;
	// 	uint32_t eid;
	// } ArrayEntry;

	// auto& edges_ = g.edges;
	auto& k_max = g.kmaxes;
	// auto& l_max = g.lmaxes;


    //edge insertion
    //for the h-index-based algorithm, both single edge and multiple edge can be processed
	vector<bool> compute(n_, false);  //needs to be computed
	// vector<bool> be_in_incore(n_, false);

	/*calculate ED value of vertices*/
	vector<uint32_t> mED(n_, 0), mPED(n_, 0);
	#pragma omp parallel for num_threads(lmax_number_of_threads)
	for (uint32_t vid = 0; vid < n_; ++vid) {
	    // for (auto neighbors: adj_in[vid]){
		for (vertex neighborIdx = 0; neighborIdx < g.in_degrees[vid]; ++neighborIdx) {
	    	vertex neighbor =  g.in_neighbors[g.in_neighbors_offset[vid] + neighborIdx];
	        if (k_max[neighbor] >= k_max[vid]) {
	            ++mED[vid];
	        }
	    }
	}
	/*calculate PED value of vertices*/
	#pragma omp parallel for num_threads(lmax_number_of_threads)
	for (uint32_t vid = 0; vid < n_; ++vid) {
	    mPED[vid] = mED[vid];
	    if(!(mED[vid] == 0)){
	        // for (auto neighbors: adj_in[vid]){
	    	for (vertex neighborIdx = 0; neighborIdx < g.in_degrees[vid]; ++neighborIdx) {
	    		vertex neighbor = g.in_neighbors[g.in_neighbors_offset[vid] + neighborIdx];
	            if(k_max[neighbor] == k_max[vid] && mED[neighbor] <= k_max[vid]){
	                --mPED[vid];
	            }
	        }
	    }
	}

	// cout << "ped done" << endl;

	vector<degree> original_kmax = k_max;

	/*find in-core of root*/
	#pragma omp parallel for num_threads(lmax_number_of_threads)
	for (auto &edge: modifiedEdges) {
	    //vector<uint32_t> dif_kmax_M_group; // the set of vertices have their kmax changed after Kmax value maintenance
	    uint32_t root = edge.first;
	    if (k_max[edge.second] < k_max[edge.first]) {
	        root = edge.second;
	    }
	    //#pragma omp parallel for num_threads(lmax_number_of_threads)
	    for(uint32_t vid = 0; vid < n_; ++vid){
	        if(k_max[vid] == original_kmax[root] && mPED[vid] > k_max[vid]){
	            compute[vid] = true;
	            //k_max[vid] = adj_in[vid].size();
	            ++k_max[vid];
	        }
	    }
	}

	// cout << "init done" << endl;

	/*do the initialization*/
	bool flag = true;
	uint32_t round_cnt = 0;
	while (flag){
	    flag = false;
	    #pragma omp parallel for num_threads(lmax_number_of_threads)
	    for(uint32_t vid = 0; vid < n_; ++vid){
	        if(compute[vid]){
	            vector<uint32_t> tmp_neighbor_in_coreness(g.in_degrees[vid],0);
	            // for(uint32_t i = 0; i < adj_in[vid].size(); ++i){
	        	for (vertex neighborIdx = 0; neighborIdx < g.in_degrees[vid]; ++neighborIdx) {
	        		vertex neighbor = g.in_neighbors[g.in_neighbors_offset[vid] + neighborIdx];
	                if(k_max[neighbor] >= original_kmax[vid]){
	                    tmp_neighbor_in_coreness[neighborIdx] = k_max[neighbor];
	                }
	            }
	            uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_in_coreness);
	            if(tmp_h_index < k_max[vid]){
	                k_max[vid] = tmp_h_index;
	                flag = true;
	            }
	        }
	    }
	    round_cnt++;
	}

	// cout << "h index done" << endl;

	return k_max;
}

void maintainKList(Graph& g, vector<pair<vertex, vertex>> modifiedEdges, degree M_) {
	const unsigned n_ = g.V;
	// const uint32_t M_ = 1;
	const int lmax_number_of_threads = 1;

	typedef struct {
		uint32_t vid;
		uint32_t eid;
	} ArrayEntry;

	auto& edges_ = g.edges;
	auto& k_max = g.kmaxes;
	auto& l_max = g.lmaxes;

	//using parallel-h-index based method to update the l_{max} value
	//std::chrono::duration<double> initialzation, find_outcore, h_index_computation;
	double initialzation = 0, find_outcore = 0, h_index_computation = 0;
	for(uint32_t k = 0 ; k <= M_ + 1; ++k){
		// cout << "one loop " << k << endl;
	    auto test1 = omp_get_wtime();
	    vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
	    vector<uint32_t> mED_out(n_, 0), mPED_out(n_, 0);
	    vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (k,0)-core
	    vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (k,0)-core
	    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
	        const uint32_t v1 = edges_[eid].first;
	        const uint32_t v2 = edges_[eid].second;
	        if(k_max[v1] >= k && k_max[v2] >= k){
	            k_adj_in[v2].push_back({v1, eid});
	            k_adj_out[v1].push_back({v2, eid});
	            if(l_max[k][v2] >= l_max[k][v1]){
	                ++mED_out[v1];
	            }
	        }
	    }

	    /*calculate PED value of vertices*/
	    #pragma omp parallel for num_threads(lmax_number_of_threads)
	    for (uint32_t vid = 0; vid < n_; ++vid) {
	        if(!k_adj_out.empty()){
	            mPED_out[vid] = mED_out[vid];
	            for (auto neighbors: k_adj_out[vid]) {
	                if(l_max[k][neighbors.vid] == l_max[k][vid] && mED_out[neighbors.vid] > l_max[k][vid]){
	                    --mPED_out[vid];
	                }
	            }
	        }
	    }

		// cout << M_ << endl;
		//
		// for (auto kmax: k_max)
		// 	cout << kmax << " ";
		// cout << endl;
		//
		// for (auto v: mED_out)
		// 	cout << v << " ";
		// cout << endl;
		// for (auto v: mPED_out)
		// 	cout << v << " ";
		// cout << endl;

	    auto test2 = omp_get_wtime();

	    vector<bool> compute(n_, false);  //needs to be computed
	    vector<bool> be_in_outcore(n_, false);
	    /*find out-core of inserted edges*/


	    #pragma omp parallel for num_threads(lmax_number_of_threads)
	    for(auto & edge : modifiedEdges){
	    	cout << "edge " << edge.first << " " << edge.second << endl;
	        if(k_max[edge.first] >= k && k_max[edge.second] >= k){
	            uint32_t root = edge.first;
	            if (l_max[k][edge.second] < l_max[k][edge.first]) {
	                root = edge.second;
	            }
	        	cout << "root " << root << endl;
	            uint32_t k_M_ = l_max[k][root];
	        	cout << "k_M_ " << k_M_ << endl;
	            for(uint32_t vid = 0; vid < n_; ++vid){
	                if(k_max[vid] >= k && l_max[k][vid] == k_M_ && mPED_out[vid] > l_max[k][vid]){
	                    compute[vid] = true;
	                    l_max[k][vid] = k_adj_out[vid].size();
	                	cout << "f " << k << "," << vid << " " << l_max[k][vid] << endl;
	                }
	            }
	        }
	    }
	    auto test3 = omp_get_wtime();


	    bool flag = true;
	    uint32_t round_cnt = 0;
	    while (flag){
	    	cout << "round_cnt " << round_cnt << endl;
	        flag = false;
	        #pragma omp parallel for num_threads(lmax_number_of_threads)
	        for(uint32_t vid = 0; vid < n_; ++vid){
	            if(compute[vid]){
	                vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(), 0);
	                for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
	                    if(l_max[k][k_adj_out[vid][i].vid] >= l_max[k][vid]){
	                        tmp_neighbor_out_coreness[i] = l_max[k][k_adj_out[vid][i].vid];
	                    }
	                }
	                uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_out_coreness);
	                if(tmp_h_index < l_max[k][vid]){
	                    l_max[k][vid] = tmp_h_index;
	                    flag = true;
	                	cout << "s " << k << "," << vid << " " << l_max[k][vid] << endl;
	                }
	            }
	        }
	        round_cnt++;
	    }

	    const auto test4 = omp_get_wtime();
	    initialzation += test2-test1;
	    find_outcore += test3-test2;
	    h_index_computation += test4-test3;
	}

	printf("Insertion lmax initilization %f ms; find out-core costs %f ms; h-index computation costs %f ms \n",
	       initialzation*1000,
	       find_outcore*1000,
	       h_index_computation*1000);

	long long width = to_string(10).length();
	ofstream outfile ("../results/maintain.txt",ios::out|ios::binary|ios::trunc);
	for (auto r: l_max) {
		for (auto v: r) {
			outfile << setw(width);
			outfile << v << " ";
		}
		outfile << "\r\n";
	}
}


int main(int argc, char *argv[]) {
	// const string filename = "../dataset/wiki_vote";
	const string filename = "../dataset/digraph2";

	auto start = chrono::steady_clock::now();

    Graph g(filename);
    cout << "> " << filename  << " V: " << g.V << " E: " << g.E << endl;

	auto res = dcore(g);
	writeDCoreResultsText(res, "../results/cudares.txt", 16);
	// writeDCoreResults(res, "../results/cudares");
	// compareDCoreResults("../results/cudares", "../results/wiki_vote");
	// compareDCoreResults("../results/cudares", "../results/amazon0601");

	// cuda maintenance,,,
	auto maintenanceStart = chrono::steady_clock::now();
	kmaxMaintenance(g, {1, 0});
	auto maintenanceEnd = chrono::steady_clock::now();
	cout << "Maintenance time: " << chrono::duration_cast<chrono::milliseconds>(maintenanceEnd - maintenanceStart).count() << "ms" << endl;
	// kmaxMaintenance(g, {1, 2});




	return 0;







	auto kmaxBefore = g.kmaxes;
	cout << "k-max-befr: ";
	for (auto v: kmaxBefore)
		cout << v << " ";
	cout << endl;

	// lets try k-max maintenance
	// vector<pair<vertex, vertex>> newEdges = {{1,0}, {6,0}, {6,3}};
	vector<pair<vertex, vertex>> newEdges = {{6,2},{6,4},{6,5},{6,7},{2,6},{4,6},{5,6},{7,6}};
	for (auto edge: newEdges) {
		g.insertEdge(edge);
		cout << "inserted edge " << edge.first << "->" << edge.second << endl;

		auto kmax = maintainKmax_(g, newEdges);

		cout << "k-max-calc: ";
		for (auto v: kmax)
			cout << v << " ";
		cout << endl;

		auto check = checkKmax(g);
		cout << "k-max-chck: ";
		for (auto v: kmax)
			cout << v << " ";
		cout << endl;

		auto changed = 0;
		for (int idx = 0; idx < kmaxBefore.size(); ++idx) {
			if (kmaxBefore[idx] != kmax[idx]) {
				changed++;
			}
		}
		cout << changed << " values changed" << endl;

		changed = 0;
		// cout << "checking...";
		for (int idx = 0; idx < kmax.size(); ++idx) {
			if (kmax[idx] != check[idx]) {
				changed++;
				cout << idx << ": " << kmax[idx] << " should be " << check[idx] << endl;
			}
		}
		if (!changed)
			cout << endl << "0 values mismatching" << endl;
		if (changed)
			cout << changed << " values mismatching" << endl;
	}

	// long long width = to_string(10).length();
	// ofstream outfile ("../results/kmaxcheck.txt",ios::out|ios::binary|ios::trunc);
	// for (auto v: kmaxBefore) {
	// 	outfile << setw(width);
	// 	outfile << v << " ";
	// }
	// outfile << "\r\n";
	// for (auto v: kmax) {
	// 	outfile << setw(width);
	// 	outfile << v << " ";
	// }
	// outfile << "\r\n";
	// for (auto v: check) {
	// 	outfile << setw(width);
	// 	outfile << v << " ";
	// }
	// outfile << "\r\n";
	// outfile.close();

	return 0;

	// l max maintenance
	// vector<pair<vertex, vertex>> newEdges = {{1,0}};
	// // vector<pair<vertex, vertex>> newEdges = {};
	// g.edges.push_back(newEdges[0]);
	//
	// // auto M = 0;
	// auto M = min(g.kmaxes[newEdges[0].first], g.kmaxes[newEdges[0].second]);
	//
	// g.kmaxes[0] = 2;
	// g.kmaxes[3] = 2;
	//
	// for (auto kmax: g.kmaxes)
	// 	cout << kmax << " ";
	// cout << endl;
	// //
	// cout << M << endl;
	// maintainKList(g, newEdges, M);

	auto end = chrono::steady_clock::now();
	cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
}
