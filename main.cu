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
#include "cuda_memory.h"
#include "graph_gen.h"
#include "cuda_utils.cuh"
#include "graph_gpu.h"
#include "utils.h"

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
		offset inEnd	= g_p.in_neighbors_offset[v] + g_p.in_degrees_orig[v];
		// offset inEnd	= g_p.in_neighbors_offset[v+1];
		offset outStart	= g_p.out_neighbors_offset[v];
		offset outEnd	= g_p.out_neighbors_offset[v] + g_p.out_degrees_orig[v];
		// offset outEnd	= g_p.out_neighbors_offset[v+1];

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

	deallocateDeviceAccessoryMemory(a_p);
	deallocateDeviceGraphMemory(g_p);

	return res;
}

vector<degree> checkKmax(Graph &g, device_graph_pointers& g_p, device_accessory_pointers& a_p) {
	// move graph to GPU
	moveGraphToDevice(g, g_p);

	// ***** calculating k-max *****
	swapInOut(g_p); // do a flip!! (we're calculating the 0 l-list)
	auto [_, kmaxes] = KList(g, g_p, a_p, 0);
	swapInOut(g_p); // let's fix the mess we made...

	return kmaxes;
}

pair<vector<degree>, vector<vector<degree>>> checkDCore(Graph &g, device_graph_pointers& g_p, device_accessory_pointers& a_p) {
	// move graph to GPU
	moveGraphToDevice(g, g_p);

	// ***** calculating k-max *****
	swapInOut(g_p); // do a flip!! (we're calculating the 0 l-list)
	auto [kmax, kmaxes] = KList(g, g_p, a_p, 0);
	swapInOut(g_p); // let's fix the mess we made...

	vector<vector<degree>> lmaxes;
	for (unsigned k = 0; k <= kmax; ++k) {
		refreshGraphOnGPU(g, g_p);	// degrees will be wrecked from previous calculation
		auto [_, core] = KList(g, g_p, a_p, k);
		lmaxes.push_back(core);
	}

	return {kmaxes, lmaxes};
}



__global__ void findMaxKmax(device_maintenance_pointers m_p, unsigned V) {
	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) break;

		atomicMax(m_p.k_max_max, m_p.k_max[v]);
	}
}
void getMaxKmax(degree& kmax, unsigned V, device_maintenance_pointers& m_p) {
	findMaxKmax<<<BLOCK_NUMS, BLOCK_DIM>>>(m_p, V);
	cudaMemcpy(&kmax, m_p.k_max_max, sizeof(degree), cudaMemcpyDeviceToHost);
}

__global__ void findMValue(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned edges) {
	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < edges; base += THREAD_COUNT) {
		unsigned e = base + global_threadIdx;
		if (e >= edges) break;

		vertex edgeFrom = g_p.modified_edges[e*2];
		vertex edgeTo = g_p.modified_edges[e*2+1];

		degree value = min(m_p.k_max[edgeFrom], m_p.k_max[edgeTo]);

		atomicMax(m_p.m_value, value);
	}
}
degree getMValue(device_graph_pointers& g_p, device_maintenance_pointers& m_p, vector<pair<vertex, vertex>>& modifiedEdges) {
	findMValue<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, modifiedEdges.size());
	degree M = 0;
	cudaMemcpy(&M, m_p.m_value, sizeof(degree), cudaMemcpyDeviceToHost);
	return M;
}


__global__ void kmaxCalculateED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) break;

		// m_p.ED[v] = 0;

		offset start = g_p.in_neighbors_offset[v];
		offset end = start + g_p.in_degrees[v];
		for (offset o = start; o < end; ++o) {
			vertex neighbor = g_p.in_neighbors[o];
			if (m_p.k_max[neighbor] >= m_p.k_max[v])
				++m_p.ED[v];
		}
	}
}

__global__ void kmaxCalculatePED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) break;

		m_p.PED[v] = m_p.ED[v];
		if (m_p.PED[v] == 0) break;

		offset start = g_p.in_neighbors_offset[v];
		offset end = start + g_p.in_degrees[v];
		for (offset o = start; o < end; ++o) {
			vertex neighbor = g_p.in_neighbors[o];
			if (m_p.k_max[neighbor] == m_p.k_max[v] && m_p.ED[neighbor] <= m_p.k_max[v])
				--m_p.PED[v];
		}
	}
}

__global__ void kmaxFindUpperBound(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	__shared__ degree root_k_max;

	if (IS_MAIN_THREAD) {
		vertex edgeFrom = g_p.modified_edges[0];
		vertex edgeTo = g_p.modified_edges[1];

		vertex root = edgeFrom;
		if (m_p.k_max[edgeTo] < m_p.k_max[edgeFrom])
			root = edgeTo;
		root_k_max = m_p.k_max[root];
	}

	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) break;

		if (m_p.k_max[v] == root_k_max && m_p.PED[v] > m_p.k_max[v]) {
			m_p.compute[v] = true;
			m_p.k_max[v]++;
		}
	}
}

__global__ void kmaxFindUpperBoundBatch(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, unsigned edges) {
	__shared__ degree root_k_max;

	// each block works on an edge at a time :)
	for (unsigned edgebase = 0; edgebase < edges; edgebase += BLOCK_NUMS) {
		unsigned e = edgebase + blockIdx.x;
		if (e >= edges) break;

		if (IS_MAIN_THREAD) {
			vertex edgeFrom = g_p.modified_edges[e*2];
			vertex edgeTo = g_p.modified_edges[e*2+1];

			vertex root = edgeFrom;
			if (m_p.k_max[edgeTo] < m_p.k_max[edgeFrom])
				root = edgeTo;
			root_k_max = m_p.k_max[root];
		}

		__syncthreads();

		for (unsigned base = 0; base < V; base += BLOCK_DIM) {
			vertex v = base + threadIdx.x;
			if (v >= V) break;

			if (m_p.k_max[v] == root_k_max && m_p.PED[v] > m_p.k_max[v]) {
				m_p.compute[v] = true;
				m_p.k_max[v]++;
			}
		}
	}
}

__global__ void kmaxRefineHIndex(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	__shared__ bool localFlag;

	if (IS_MAIN_THREAD) localFlag = false;
	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;

		if (v >= V) break;
		if (!m_p.compute[v]) continue;

		offset histogramStart = g_p.in_neighbors_offset[v] + v;
		offset histogramEnd = histogramStart + m_p.k_max[v];
		for (offset o = histogramStart; o <= histogramEnd; ++o)
			m_p.histograms[o] = 0;

		offset start = g_p.in_neighbors_offset[v];
		offset end = start + g_p.in_degrees[v];
		for (offset o = start; o < end; ++o) {
			vertex neighbor = g_p.in_neighbors[o];

			m_p.histograms[histogramStart + min(m_p.k_max[v],m_p.k_max[neighbor])]++;
		}

		// calculate h index
		degree tmp_h_index = hOutIndex(m_p, v, g_p.in_neighbors_offset[v], m_p.k_max[v]);

		if (tmp_h_index < m_p.k_max[v]) {
			m_p.k_max[v] = tmp_h_index;
			localFlag = true;
		}
	}

	__syncthreads();
	if (IS_MAIN_THREAD && localFlag)
		*m_p.flag = true;
}

// we expect modified edges to already be set in device graph pointers!!
// we also expect kmax on gpu to be "in place" as in, we wont load it from graphdata or place it into graphdata
void kmaxMaintenance(GraphData g, device_graph_pointers& g_p, device_maintenance_pointers& m_p, vector<pair<vertex, vertex>>& modifiedEdges) {
	// cout << "\tLoading graph on GPU" << endl;
	// move graph to GPU
	// moveGraphToDevice(g, g_p); // todo: this shouldnt be here for final maintenance

	#ifdef PRINT_STEPS
	cout << "K MAX for batch {";
	for (auto& edge: modifiedEdges) {
		cout << edge.first << "->" << edge.second;
		if (edge != modifiedEdges.back())
			cout << ", ";
	}
	cout << "}" << endl;
	#endif


	// cout << "\tSetting up maintenance CUDA memory" << endl;
	initializeDeviceMaintenanceMemory(g.V, m_p);

	// cout << "\tCalculating ED and PED" << endl;
	kmaxCalculateED<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V);
	kmaxCalculatePED<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V);
	#ifdef PRINT_STEPS
	cout << "             ";
	for (vertex v = 0; v < g.V; ++v)
		cout << v << " ";
	cout << endl;
	vector<degree> ED(g.V);
	vector<degree> PED(g.V);
	cudaMemcpy(ED.data(), m_p.ED, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	cudaMemcpy(PED.data(), m_p.PED, g.V * sizeof(degree), cudaMemcpyDeviceToHost);

	cout << "         ED: ";
	for (auto v: ED)
		cout << v << " ";
	cout << endl;
	cout << "        PED: ";
	for (auto v: PED)
		cout << v << " ";
	cout << endl;
	#endif

	// cout << "\tCalculating upper bounds" << endl;
	// we would load modified edges here if they werent already loaded
	if (modifiedEdges.size() > 1)
		kmaxFindUpperBoundBatch<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V, modifiedEdges.size());
	else
		kmaxFindUpperBound<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V);

	#ifdef PRINT_STEPS
	vector<degree> upper(g.V);
	cudaMemcpy(upper.data(), m_p.k_max, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "   kmax_upp: ";
	for (auto v: upper)
		cout << v << " ";
	cout << endl;
	vector<unsigned> compute(g.V);
	cudaMemcpy(compute.data(), m_p.compute, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "compute    : ";
	for (auto v: compute)
		cout << v << " ";
	cout << endl;
	#endif


	bool flag = true;
	while (flag) {
		cudaMemset(m_p.flag, false, sizeof(bool));
		kmaxRefineHIndex<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V);
		cudaMemcpy(&flag, m_p.flag, sizeof(bool), cudaMemcpyDeviceToHost);
	}

	// load back into cpu
	// getKmaxFromDeviceMemory(m_p, g.kmaxes);

	#ifdef PRINT_STEPS
	vector<degree> kmax_refine(g.V);
	cudaMemcpy(kmax_refine.data(), m_p.k_max, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "kmax_refine: ";
	for (auto v: kmax_refine)
		cout << v << " ";
	cout << endl << endl;
	#endif
}


__global__ void kListCalculateED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) break;

		// m_p.ED[v] = 0;
		if (m_p.k_max[v] < k) continue;

		offset start = g_p.out_neighbors_offset[v];
		offset end = start + g_p.out_degrees[v];
		for (offset o = start; o < end; ++o) {
			vertex neighbor = g_p.out_neighbors[o];

			if (m_p.k_max[neighbor] < k) continue;

			// we need to maintain some sort of k_adj_in/out list here?

			if (m_p.l_max[neighbor] >= m_p.l_max[v])
				++m_p.ED[v];
		}
	}

}

__global__ void kListCalculatePED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) break;

		m_p.PED[v] = m_p.ED[v];
		if (m_p.PED[v] == 0) continue;
		if (m_p.k_max[v] < k) continue;

		offset start = g_p.out_neighbors_offset[v];
		offset end = start + g_p.out_degrees[v];
		for (offset o = start; o < end; ++o) {
			vertex neighbor = g_p.out_neighbors[o];

			if (m_p.k_max[neighbor] < k) continue;

			if (m_p.l_max[neighbor] == m_p.l_max[v] && m_p.ED[neighbor] <= m_p.l_max[v])
				--m_p.PED[v];
		}
	}
}

__global__ void kListFindUpperBound(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	// __shared__ degree root_l_max;

	if (IS_MAIN_THREAD) {
		// vertex edgeFrom = g_p.modified_edges[0];
		// vertex edgeTo = g_p.modified_edges[1];

		// vertex root = edgeFrom;
		// if (m_p.l_max[edgeTo] < m_p.l_max[edgeFrom])
		// 	root = edgeTo;
		// root_l_max = m_p.l_max[root];
	}

	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) break;

		// if (m_p.k_max[v] >= k && m_p.l_max[v] <= root_l_max + 1 && m_p.PED[v] >= m_p.l_max[v]) {
		// if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max && m_p.PED[v] > m_p.l_max[v]) {
		// if (m_p.k_max[v] >= k && m_p.PED[v] > m_p.l_max[v]) {
		// if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max) {
		if (m_p.k_max[v] >= k) {	// todo: this is pathetic... its rly just redoing the decomposition...
			// we only want to compute upper bound once
			if (atomicTestAndSet(&m_p.compute[v])) {

				// m_p.l_max[v] = g_p.out_degrees[v];
				// m_p.l_max[v] = k_adj_out[v].size();

				// // todo: this is probably rly expensive
				unsigned k_adj_out_v = 0;
				offset start = g_p.out_neighbors_offset[v];
				offset end = start + g_p.out_degrees[v];
				for (offset o = start; o < end; ++o) {
					vertex neighbor = g_p.out_neighbors[o];

					if (m_p.k_max[neighbor] < k) continue;

					k_adj_out_v++;
				}
				m_p.l_max[v] = k_adj_out_v;
			}
		}
	}
}

__global__ void kListFindUpperBoundBatch(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, unsigned edges, degree k) {
	__shared__ bool skip_edge;
	// __shared__ degree root_l_max;

	// each block works on an edge at a time :)
	for (unsigned edgebase = 0; edgebase < edges; edgebase += BLOCK_NUMS) {
		unsigned e = edgebase + blockIdx.x;
		if (e >= edges) break;

		if (IS_MAIN_THREAD) {
			// vertex edgeFrom = g_p.modified_edges[e*2];
			// vertex edgeTo = g_p.modified_edges[e*2+1];

			// skip_edge = (m_p.k_max[edgeFrom] < k || m_p.k_max[edgeTo] < k);	// is this right?!?
			skip_edge = false;

			// vertex root = edgeFrom;
			// if (m_p.l_max[edgeTo] < m_p.l_max[edgeFrom])
			// 	root = edgeTo;
			// root_l_max = m_p.l_max[root];
		}

		__syncthreads();
		if (skip_edge) continue;

		for (unsigned base = 0; base < V; base += BLOCK_DIM) {
			vertex v = base + threadIdx.x;
			if (v >= V) break;

			// if (m_p.k_max[v] >= k && m_p.l_max[v] <= root_l_max + 1 && m_p.PED[v] >= m_p.l_max[v]) {
			// if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max && m_p.PED[v] > m_p.l_max[v]) {
			// if (m_p.k_max[v] >= k && m_p.PED[v] > m_p.l_max[v]) {
			// if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max) {
			if (m_p.k_max[v] >= k) {	// todo: this is pathetic... its rly just redoing the decomposition...
				// we only want to compute upper bound once
				if (atomicTestAndSet(&m_p.compute[v])) {

					// m_p.l_max[v] = g_p.out_degrees[v];
					// m_p.l_max[v] = k_adj_out[v].size();

					// // todo: this is probably rly expensive
					unsigned k_adj_out_v = 0;
					offset start = g_p.out_neighbors_offset[v];
					offset end = start + g_p.out_degrees[v];
					for (offset o = start; o < end; ++o) {
						vertex neighbor = g_p.out_neighbors[o];

						if (m_p.k_max[neighbor] < k) continue;

						k_adj_out_v++;
					}
					m_p.l_max[v] = k_adj_out_v;
				}
			}
		}
	}
}

#ifdef HINDEX_NOWARP
__global__ void kListRefineHIndex(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	__shared__ bool localFlag;

	if (IS_MAIN_THREAD)
		localFlag = false;
	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;

		if (v >= V) break;
		if (!m_p.compute[v]) continue;

		// if (m_p.k_max[v] < k) continue; // not necessary since this vertex would never be computed anyway

		// make histogram bucket (V+E) initialized to zero
		offset histogramStart = g_p.out_neighbors_offset[v] + v;
		offset histogramEnd = histogramStart + m_p.l_max[v];
		for (offset o = histogramStart; o <= histogramEnd; ++o)
			m_p.histograms[o] = 0;

		// put in neighbors into buckets
		offset inStart = g_p.in_neighbors_offset[v];
		offset inEnd = inStart + g_p.in_degrees[v];
		for (offset o = inStart; o < inEnd; ++o) {
			vertex neighbor = g_p.in_neighbors[o];

			if (m_p.k_max[neighbor] < k) continue;

			// each vertex write to histogram buffer at min(lmax[v], lmax[neighbor])
			m_p.histograms[histogramStart + min(m_p.l_max[v],m_p.l_max[neighbor])]++;
		}


		// kernel that scans back from historgram[lmax[v]] and adds until the value is bigger or equal to k, which then returns the index at which we got above or equal to k. And that is h+ index
		degree tmp_h_in_index = hInIndex(m_p, v, g_p.out_neighbors_offset[v], m_p.l_max[v], k);


		// reinitialize histograms to 0
		for (offset o = histogramStart; o <= histogramEnd; ++o)
			m_p.histograms[o] = 0;

		offset outStart = g_p.out_neighbors_offset[v];
		offset outEnd = outStart + g_p.out_degrees[v];
		for (offset o = outStart; o < outEnd; ++o) {
			vertex neighbor = g_p.out_neighbors[o];

			if (m_p.k_max[neighbor] < k) continue;

			m_p.histograms[histogramStart + min(m_p.l_max[v],m_p.l_max[neighbor])]++;
		}

		degree tmp_h_out_index = hOutIndex(m_p, v, g_p.out_neighbors_offset[v], m_p.l_max[v]);

		degree res = min(tmp_h_out_index, tmp_h_in_index);

		if (res < m_p.l_max[v]) {
			m_p.new_l_max[v] = res;
			localFlag = true;
		}
	}

	__syncthreads();
	if (IS_MAIN_THREAD && localFlag)
		*m_p.flag = true;
}
#endif

#ifdef HINDEX_WARP
__global__ void kListRefineHIndex(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	__shared__ bool localFlag;

	if (IS_MAIN_THREAD)
		localFlag = false;
	__syncthreads();

	for (unsigned base = 0; base < V; base += WARP_COUNT) {
		// each warp has its own vertex
		vertex v = base + blockIdx.x * WARPS_EACH_BLOCK + WARP_ID;

		if (v >= V) break;
		if (!m_p.compute[v]) continue;

		// if (m_p.k_max[v] < k) continue; // not necessary since this vertex would never be computed anyway

		// make histogram bucket (V+E) initialized to zero
		offset histogramStart = g_p.out_neighbors_offset[v] + v;
		offset histogramEnd = histogramStart + m_p.l_max[v];
		for (offset o = histogramStart; o <= histogramEnd; o += WARP_SIZE) {
			if (o + LANE_ID <= histogramEnd) m_p.histograms[o + LANE_ID] = 0;
		}
		// if (IS_MAIN_IN_WARP) {
		// 	for (offset o = histogramStart; o <= histogramEnd; ++o)
		// 		m_p.histograms[o] = 0;
		// }
		__syncwarp();

		// put in neighbors into buckets
		offset inStart = g_p.in_neighbors_offset[v];
		offset inEnd = inStart + g_p.in_degrees[v];
		for (offset o = inStart; o < inEnd; o += WARP_SIZE) {
			if (o + LANE_ID >= inEnd) continue;
			vertex neighbor = g_p.in_neighbors[o + LANE_ID];

			if (m_p.k_max[neighbor] < k) continue;

			// each vertex write to histogram buffer at min(lmax[v], lmax[neighbor])
			atomicAdd(m_p.histograms + (histogramStart + min(m_p.l_max[v],m_p.l_max[neighbor])), 1);
		}
		// if (IS_MAIN_IN_WARP) {
		// 	for (offset o = inStart; o < inEnd; ++o) {
		// 		vertex neighbor = g_p.in_neighbors[o];
		//
		// 		if (m_p.k_max[neighbor] < k) continue;
		//
		// 		// each vertex write to histogram buffer at min(lmax[v], lmax[neighbor])
		// 		m_p.histograms[histogramStart + min(m_p.l_max[v],m_p.l_max[neighbor])]++;
		// 	}
		// }
		__syncwarp();

		// kernel that scans back from historgram[lmax[v]] and adds until the value is bigger or equal to k, which then returns the index at which we got above or equal to k. And that is h+ index
		degree tmp_h_in_index = 0;
		if (IS_MAIN_IN_WARP)
			tmp_h_in_index = hInIndex(m_p, v, g_p.out_neighbors_offset[v], m_p.l_max[v], k);
		__syncwarp();

		// reinitialize histograms to 0
		for (offset o = histogramStart; o <= histogramEnd; o += WARP_SIZE) {
			if (o + LANE_ID <= histogramEnd) m_p.histograms[o + LANE_ID] = 0;
		}
		// if (IS_MAIN_IN_WARP) {
		// 	for (offset o = histogramStart; o <= histogramEnd; ++o)
		// 		m_p.histograms[o] = 0;
		// }
		__syncwarp();

		offset outStart = g_p.out_neighbors_offset[v];
		offset outEnd = outStart + g_p.out_degrees[v];
		for (offset o = outStart; o < outEnd; o += WARP_SIZE) {
			if (o + LANE_ID >= outEnd) continue;
			vertex neighbor = g_p.out_neighbors[o + LANE_ID];

			if (m_p.k_max[neighbor] < k) continue;

			atomicAdd(m_p.histograms + (histogramStart + min(m_p.l_max[v],m_p.l_max[neighbor])), 1);
		}
		// if (IS_MAIN_IN_WARP) {
		// 	for (offset o = outStart; o < outEnd; ++o) {
		// 		vertex neighbor = g_p.out_neighbors[o];
		//
		// 		if (m_p.k_max[neighbor] < k) continue;
		//
		// 		m_p.histograms[histogramStart + min(m_p.l_max[v],m_p.l_max[neighbor])]++;
		// 	}
		// }
		__syncwarp();


		if (IS_MAIN_IN_WARP) {
			degree tmp_h_out_index = hOutIndex(m_p, v, g_p.out_neighbors_offset[v], m_p.l_max[v]);

			degree res = min(tmp_h_out_index, tmp_h_in_index);

			if (res < m_p.l_max[v]) {
				m_p.new_l_max[v] = res;
				localFlag = true;
			}
		}
	}

	__syncthreads();
	if (IS_MAIN_THREAD && localFlag)
		*m_p.flag = true;
}
#endif

void kListMaintenance(GraphData g, device_graph_pointers& g_p, device_maintenance_pointers& m_p, vector<pair<vertex, vertex>>& modifiedEdges, degree k) {
	#ifdef PRINT_STEPS
		cout << "L MAX for k = " << k << endl;;
	#endif

	initializeDeviceMaintenanceMemory(g.V, m_p);

	kListCalculateED<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V, k);
	kListCalculatePED<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V, k);
	#ifdef PRINT_STEPS
	cout << "             ";
	for (vertex v = 0; v < g.V; ++v)
		cout << v << " ";
	cout << endl;
	vector<degree> ED(g.V);
	vector<degree> PED(g.V);
	cudaMemcpy(ED.data(), m_p.ED, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	cudaMemcpy(PED.data(), m_p.PED, g.V * sizeof(degree), cudaMemcpyDeviceToHost);

	cout << "         ED: ";
	for (auto v: ED)
		cout << v << " ";
	cout << endl;
	cout << "        PED: ";
	for (auto v: PED)
		cout << v << " ";
	cout << endl;
	#endif

	if (modifiedEdges.size() > 1)
		kListFindUpperBoundBatch<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V, modifiedEdges.size(), k);
	else
		kListFindUpperBound<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V, k);

	#ifdef PRINT_STEPS
	vector<degree> upper(g.V);
	cudaMemcpy(upper.data(), m_p.l_max, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "lmax["<<k<<"]_upp: ";
	for (auto v: upper)
		cout << v << " ";
	cout << endl;
	vector<unsigned> compute(g.V);
	cudaMemcpy(compute.data(), m_p.compute, g.V * sizeof(unsigned), cudaMemcpyDeviceToHost);
	cout << "compute    : ";
	for (auto v: compute)
		cout << v << " ";
	cout << endl;
	unsigned count = 0;
	#endif

	bool flag = true;
	while (flag) {
		cudaMemset(m_p.flag, false, sizeof(bool));

		cudaMemcpy(m_p.new_l_max, m_p.l_max, g.V * sizeof(degree), cudaMemcpyDeviceToDevice);
		kListRefineHIndex<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V, k);
		cudaMemcpy(m_p.l_max, m_p.new_l_max, g.V * sizeof(degree), cudaMemcpyDeviceToDevice);

		cudaMemcpy(&flag, m_p.flag, sizeof(bool), cudaMemcpyDeviceToHost);

		#ifdef PRINT_STEPS
		cout << "round_cnt: " << count++ << endl;
		vector<degree> new_l_max(g.V);
		cudaMemcpy(new_l_max.data(), m_p.l_max, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
		cout << "  lmax["<<k<<"] =  ";
		for (vertex v = 0; v < g.V; ++v)
			cout << new_l_max[v] << " ";
		cout << endl;
		#endif
	}

	#ifdef PRINT_STEPS
	cout << endl;
	#endif
}


vector<vector<pair<vertex, vertex>>> getEdgeBatches(GraphData g, const vector<pair<vertex, vertex>>& edgesToBeInserted) {
	vector<vector<pair<vertex, vertex>>> batches;

	degree maxKmax = 0;
	vector<degree> edgeRoot(edgesToBeInserted.size());
	vector<degree> edgeKmax(edgesToBeInserted.size());
	for (unsigned eid = 0; eid < edgesToBeInserted.size(); ++eid) {
		auto edge = edgesToBeInserted[eid];
		vertex root = edge.second;
		if (g.kmaxes[edge.first] > g.kmaxes[edge.second])
			root = edge.first;
		edgeRoot[eid] = root;
		edgeKmax[eid] = g.kmaxes[root];
		maxKmax = max(maxKmax, g.kmaxes[root]);
	}
	// buckets
	vector<vector<unsigned>> B(maxKmax + 1);
	for (unsigned eid = 0; eid < edgesToBeInserted.size(); ++eid) {
		B[maxKmax - edgeKmax[eid]].push_back(eid);
	}

	for (auto& batch: B) { // each batch is a list of edges with the same kmax value
		if (batch.empty()) continue;

		bool flag = true;
		while (flag) {
			vector<pair<vertex, vertex>> candidateBatch;
			vector<bool> v_ (g.V, false);

			for (auto it = batch.begin(); it != batch.end();) {
				unsigned eid = *(it);
				if (candidateBatch.empty() || !v_[edgeRoot[eid]]) {
					candidateBatch.push_back(edgesToBeInserted[eid]);
					v_[edgeRoot[eid]] = true;
					it = batch.erase(it);
				} else {
					++it;
				}
			}

			if (!candidateBatch.empty())
				batches.push_back(candidateBatch);
			else
				flag = false;	// we're done
		}
	}

	return batches;
}


template<typename F>
chrono::duration<double, milli> funcTime(F&& func){
	auto t1 = chrono::high_resolution_clock::now();
	func();
	auto t2 = chrono::high_resolution_clock::now();
	return t2- t1;
}

void maintenance(Graph& g, device_graph_pointers& g_p, device_maintenance_pointers& m_p, vector<pair<vertex, vertex>>& edgesToAdd, bool inPlace = false) {
	chrono::duration<double, milli> insertTime{}, kmaxTime{}, klistTime{};

	auto batches = getEdgeBatches({g.V, g.kmaxes, g.lmaxes}, edgesToAdd);
	#ifdef PRINT_MAINTENANCE_STATS
	cout << "batches: " << batches.size() << " ";
	if (!batches.empty()) {
		cout << "max(" << (*max_element(batches.begin(), batches.end(), [](const vector<pair<vertex, vertex>>& a, const vector<pair<vertex, vertex>>& b){return a.size() < b.size();})).size() << ") ";
		unsigned long long avg = 0;
		for (auto& batch: batches) avg += batch.size();
		avg /= batches.size();
		cout << "avg(" << avg << ") ";
		cout << "one(" << count_if(batches.begin(), batches.end(), [](const vector<pair<vertex, vertex>>& a){return a.size() == 1;}) << ") ";
	}
	#endif

	degree M = 0;
	for (auto& batch: batches) {
		if (batch.empty()) continue;

		// getKmaxFromDeviceMemory(m_p, g.kmaxes);
		// for (auto edge: batch) {
		// 	M = max(M, min(g.kmaxes[edge.first], g.kmaxes[edge.second]));
		// }
		insertTime += funcTime([&] {
			putModifiedEdgesInDeviceMemory(g_p, batch);
			if (inPlace) g.insertEdgesInPlace(batch);
			else {
				assert(FORCE_REBUILD_GRAPH);
				g.insertEdges(batch);
			}
			moveGraphToDevice(g, g_p);
		});
		kmaxTime += funcTime([&] {
			putKmaxInDeviceMemory(m_p, g.kmaxes);
			M = max(M, getMValue(g_p, m_p,batch));
			kmaxMaintenance({g.V, g.kmaxes, g.lmaxes}, g_p, m_p, batch);
			getKmaxFromDeviceMemory(m_p, g.kmaxes);
		});
	}

	getMaxKmax(g.kmax, g.V, m_p);

	if (g.kmax+1 != g.lmaxes.size()) {
		unsigned prevSize = g.lmaxes.size();
		g.lmaxes.resize(g.kmax+1);

		for (unsigned i = prevSize; i < g.kmax+1; ++i)
			g.lmaxes[i] = vector<degree>(g.V);
	}

	putModifiedEdgesInDeviceMemory(g_p, edgesToAdd);
	if (M > g.kmax-1) M = g.kmax-1;
	#ifdef PRINT_MAINTENANCE_STATS
	cout << "M+1=" << M+1 << " ";
	#endif
	klistTime = funcTime([&] {
		for (degree k = 0; k <= M+1; ++k) {
			putLmaxInDeviceMemory(m_p, g.lmaxes[k]);
			kListMaintenance({g.V, g.kmaxes, g.lmaxes}, g_p, m_p, edgesToAdd, k);
			getLmaxFromDeviceMemory(m_p, g.lmaxes[k]);
		}
	});

	#ifdef PRINT_MAINTENANCE_STATS
	cout << "insert: " << insertTime << " kmax: " << kmaxTime << " klist: " << klistTime << endl;
	#endif
}
void maintenance(GraphGPU& g, device_graph_pointers& g_p, device_maintenance_pointers& m_p, vector<pair<vertex, vertex>>& edgesToAdd) {
	chrono::duration<double, milli> insertTime{}, kmaxTime{}, klistTime{};

	auto batches = getEdgeBatches({g.V, g.kmaxes, g.lmaxes}, edgesToAdd);
	#ifdef PRINT_MAINTENANCE_STATS
	cout << "batches: " << batches.size() << " ";
	if (!batches.empty())
		cout << "max(" << (*max_element(batches.begin(), batches.end(), [](const vector<pair<vertex, vertex>>& a, const vector<pair<vertex, vertex>>& b){return a.size() < b.size();})).size() << ") ";
	#endif

	degree M = 0;
	for (auto& batch: batches) {
		if (batch.empty()) continue;

		insertTime += funcTime([&] {
			putModifiedEdgesInDeviceMemory(g_p, batch);
			g.insertEdges(batch);
			// moveGraphToDevice(g, g_p);
		});
		M = max(M, getMValue(g_p, m_p,batch));
		kmaxTime += funcTime([&] {kmaxMaintenance({g.V, g.kmaxes, g.lmaxes}, g_p, m_p, batch);});
	}

	getMaxKmax(g.kmax, g.V, m_p);

	if (g.kmax+1 != g.lmaxes.size()) {
		unsigned prevSize = g.lmaxes.size();
		g.lmaxes.resize(g.kmax+1);

		for (unsigned i = prevSize; i < g.kmax+1; ++i)
			g.lmaxes[i] = vector<degree>(g.V);
	}

	putModifiedEdgesInDeviceMemory(g_p, edgesToAdd);
	if (M > g.kmax-1) M = g.kmax-1;
	klistTime = funcTime([&] {
		for (degree k = 0; k <= M+1; ++k) {
			putLmaxInDeviceMemory(m_p, g.lmaxes[k]);
			kListMaintenance({g.V, g.kmaxes, g.lmaxes}, g_p, m_p, edgesToAdd, k);
			getLmaxFromDeviceMemory(m_p, g.lmaxes[k]);
		}
	});

	#ifdef PRINT_MAINTENANCE_STATS
	cout << "insert: " << insertTime << " kmax: " << kmaxTime << " klist: " << klistTime << endl;
	#endif
}


vector<vector<pair<vertex, vertex>>> getEdgeBatch(
        const vector<pair<vertex, vertex>>& edges_to_be_modified,
        const  vector<vector<pair<vertex,vertex>>>& old_d_core_decomposition,
        vector<pair<vertex,vertex>>& remaining_unbatched_edges) {

    remaining_unbatched_edges.clear();
    vector<vector<pair<uint32_t, uint32_t>>> generated_edge_batch;
    /*initializations*/
    uint32_t max_k_max = 0, max_vertex_id = 0;     //maximal edge k_max value, maximal vertex id in the inserted/deleted edges
    for(const auto &e: edges_to_be_modified){
            max_vertex_id = std::max(max_vertex_id, std::max(e.first, e.second));
    }
    uint32_t vec_bound = max_vertex_id + 1;

	typedef struct {
		uint32_t vid;
		uint32_t kmax;
	} ArrayEntry;

    vector<ArrayEntry> edge_k_max(edges_to_be_modified.size()); //k_max value of inserted/deleted edges
    #pragma omp parallel for num_threads(32)
    for(int eid = 0; eid < edges_to_be_modified.size(); ++eid){
        auto e = edges_to_be_modified[eid];
        if(old_d_core_decomposition[e.first][0].first < old_d_core_decomposition[e.second][0].first){
            edge_k_max[eid].vid = e.first;
            edge_k_max[eid].kmax = old_d_core_decomposition[e.first][0].first;
        }
        else{
            edge_k_max[eid].vid = e.second;
            edge_k_max[eid].kmax = old_d_core_decomposition[e.second][0].first;
        }
    }

    for(int eid = 0; eid < edges_to_be_modified.size(); ++eid){
        max_k_max = max(max_k_max, edge_k_max[eid].kmax);
    }

    vector<vector<int>> B(max_k_max + 1); // Empty buckets
    for (int eid = 0; eid < edges_to_be_modified.size(); ++eid) {
        B[edge_k_max[eid].kmax].push_back(eid);
    }

    //remove empty buckets
    B.erase (std::remove_if (B.begin (), B.end (), [] (const auto& vv)
    {
        return vv.empty ();
    }), B.end ());

    vector<vector<pair<uint32_t, uint32_t>>> batches_kmax_hierarchy, batches_kedge_set; //batches of edges,
    uint32_t k_max_hierarchy_size = 0, generated_edge_batch_size = 0;

	/*process remaining dis-batched B with edges has same k_max value*/
    for(auto &batch : B){        //each batch is a list of edges with same k_max value
        bool flag = true;
        while (flag){
            //vector<int> candidate_batch;
            vector<pair<uint32_t,uint32_t>> candidate_batch_edge;
            vector<bool> v_ (vec_bound, false);
            for(auto it = batch.begin(); it != batch.end(); ){
                int eid = *(it);
                if(candidate_batch_edge.empty() || !v_[edge_k_max[eid].vid]) {
                    //candidate_batch.push_back(eid);
                    candidate_batch_edge.push_back(edges_to_be_modified[eid]);
                    v_[edge_k_max[eid].vid] = true;
                    it = batch.erase(it);
                }
                else{
                    ++it;
                }
            }
            uint32_t batch_size = candidate_batch_edge.size();
            if(batch_size > 1) {    //avoid single edge as batch
                batches_kedge_set.push_back(candidate_batch_edge);
                k_max_hierarchy_size += candidate_batch_edge.size();
            }
            else if(batch_size == 1){     //single edge as a batch
                remaining_unbatched_edges.push_back(/*edges_to_be_modified[candidate_batch[0]]*/candidate_batch_edge[0]);   //sequential processing
                flag = false;
            }
            else{
                flag = false;
            }
        }
    }
    generated_edge_batch.insert(generated_edge_batch.end(), batches_kedge_set.begin(), batches_kedge_set.end());


    for(auto &batch : B){
        if(!batch.empty()){
            for(auto &eid : batch){
                remaining_unbatched_edges.push_back(edges_to_be_modified[eid]);
            }
        }
    }

    return generated_edge_batch;
}

int main(int argc, char *argv[]) {
	// generateGraph("../dataset/random", 10, 40);

	// const string filename = "../dataset/digraph";
	// const string filename = "../dataset/wiki_vote";
	const string filename = "../dataset/email";
	// const string filename = "../dataset/live_journal";

	auto start = chrono::steady_clock::now();

    Graph g(filename);
    cout << "> " << filename  << " V: " << g.V << " E: " << g.E << " AVG_DEG: " << g.E / g.V << endl;


	if (!readDecompFile(g, filename)) {
		cout << "Calculating decomp file..." << endl;
		auto res = dcore(g);
		writeDecompFile(g, filename);
	}

	// writeDCoreResultsText(res, "../results/cudares.txt", 16);
	// writeDCoreResults(res, "../results/cudares");
	// compareDCoreResults("../results/cudares", "../results/wiki_vote");
	// compareDCoreResults("../results/cudares", "../results/amazon0601");

	device_graph_pointers g_p;
	unsigned graph_mem_size = allocateDeviceGraphMemory(g, g_p);
	device_accessory_pointers a_p;
	unsigned accessory_mem_size = allocateDeviceAccessoryMemory(g, a_p);
	device_maintenance_pointers m_p;
	unsigned maintenance_mem_size = allocateDeviceMaintenanceMemory(g, m_p);
	cout << "Allocated memory\t\t" << calculateSize(graph_mem_size + accessory_mem_size + maintenance_mem_size) << "\tgraph: " << calculateSize(graph_mem_size) << " accessory: " << calculateSize(accessory_mem_size) << " maintenance: " << calculateSize(maintenance_mem_size) << endl;

#define SINGLE_INSERT
// #define SPEED_TEST
// #define STEP_TEST

	// ***********************************************r*******************************
#ifdef SINGLE_INSERT
	assert(OFFSET_GAP >= 1);	// gapsize needs to be atleast 1;
	// vector<pair<vertex, vertex>> edgeToAdd = {{0, 1}};	// digraph
	// vector<pair<vertex, vertex>> edgeToAdd = {{456, 316}};	// wiki_vote
	// vector<pair<vertex, vertex>> edgeToAdd = {{1, 0}};	// email
	// vector<pair<vertex, vertex>> edgeToAdd = {{50, 23}};	// email
	vector<pair<vertex, vertex>> edgeToAdd = {{47, 46}};	// email
	// vector<pair<vertex, vertex>> edgeToAdd = {{2, 1}};	// live_journal
	// vector<pair<vertex, vertex>> edgeToAdd = {};
	maintenance(g, g_p, m_p, edgeToAdd, true);

	auto comp = checkDCore(g, g_p, a_p);
	if (!SINGlE_INSERT_SKIP_CHECK) {
		auto errors = 0;
		for (vertex v = 0; v < g.V; ++v) {
			if (g.kmaxes[v] != comp.first[v]) {
				cout << "mismatching kmax["<<v<<"] " << g.kmaxes[v] << "!=" << comp.first[v] << endl;
				errors++;
			}
		}
		cout << "kmax check done - found " << errors << " errors" << endl;
		errors = 0;
		// cout << "skipping lmax" << endl;
		for (degree k = 0; k < g.lmaxes.size(); ++k) {
			for (vertex v = 0; v < g.V; ++v) {
				if (g.lmaxes[k][v] != comp.second[k][v]) {
					cout << "mismatching lmax["<<k<<"]["<<v<<"] " << g.lmaxes[k][v] << "!=" << comp.second[k][v] << endl;
					errors++;
				}
			}
		}
		cout << "lmax check done - found " << errors << " errors" << endl;
	}
#endif

	// ***********************************************r*******************************
#ifdef SPEED_TEST
	GraphGPU m_gpu(g, g_p);
	// Graph m_gpu(g.V);
	unsigned batchSize = 1;
	unsigned edgesAdded = 0;
	for (unsigned batchStart = 0; batchStart < g.E; batchStart += batchSize) {
		vector<pair<vertex, vertex>> edgesToAdd;
		for (unsigned eid = batchStart; eid < batchStart + batchSize && eid < g.E; ++eid) {
			edgesAdded++;
			edgesToAdd.push_back(g.edges[eid]);
		}
		if (batchSize > 1)				cout << edgesAdded << "/" << g.E << endl;
		else if (edgesAdded % 100 == 0) cout << edgesAdded << "/" << g.E << endl;

		maintenance(m_gpu, g_p, m_p, edgesToAdd);
	}

	for (vertex v = 0; v < m_gpu.V; ++v) {
		if (m_gpu.kmaxes[v] != g.kmaxes[v]) {
			cout << edgesAdded << " has mismatching kmax["<<v<<"] " << m_gpu.kmaxes[v] << "!=" << g.kmaxes[v] << endl;
		}
	}
	if (m_gpu.lmaxes.size() != g.lmaxes.size()) {
		cout << edgesAdded << " has mismatching k-max sizes! " << m_gpu.lmaxes.size() << "!=" << g.lmaxes.size() << endl;
	}
	for (degree k = 0; k < g.lmaxes.size(); ++k) {
		for (vertex v = 0; v < m_gpu.V; ++v) {
			if (m_gpu.lmaxes[k][v] != g.lmaxes[k][v]) {
				cout << edgesAdded << " has mismatching lmax["<<k<<"]["<<v<<"] " << m_gpu.lmaxes[k][v] << "!=" << g.lmaxes[k][v] << endl;
			}
		}
	}
	cout << "done" << endl;
#endif

	// ******************************************************************************
#ifdef STEP_TEST
	Graph m(g.V);
	unsigned batchSize = 1000;
	unsigned edgesAdded = 0;
	unsigned errors = 0;
	for (unsigned batchStart = 0; batchStart < g.E; batchStart += batchSize) {
		vector<pair<vertex, vertex>> edgesToAdd;
		for (unsigned eid = batchStart; eid < batchStart + batchSize && eid < g.E; ++eid) {
			edgesAdded++;
			edgesToAdd.push_back(g.edges[eid]);
		}
		if (batchSize > 1)				cout << edgesAdded << "/" << g.E << endl;
		else if (edgesAdded % 100 == 0) cout << edgesAdded << "/" << g.E << endl;

		bool hasErrors = false;

		maintenance(m, g_p, m_p, edgesToAdd);
		auto actual_dcore = checkDCore(m, g_p, a_p);

		// cout << "kmax: ";
		// for (auto kmax: m.kmaxes)
		// 	cout << kmax << " ";
		// cout << endl;
		// cout << "actu: ";
		// for (auto kmax: checkKmax(m, g_p, a_p))
		// 	cout << kmax << " ";
		// cout << endl;


		for (vertex v = 0; v < m.V; ++v) {
			if (m.kmaxes[v] != actual_dcore.first[v]) {
				cout << edgesAdded << " has mismatching kmax["<<v<<"] " << m.kmaxes[v] << "!=" << actual_dcore.first[v] << endl;
				errors++;
			}
		}
		if (m.lmaxes.size() != actual_dcore.second.size()) {
			cout << edgesAdded << " has mismatching k-max sizes! " << m.lmaxes.size() << "!=" << actual_dcore.second.size() << endl;
			continue;
		}
		for (degree k = 0; k < actual_dcore.second.size(); ++k) {
			for (vertex v = 0; v < m.V; ++v) {
				if (m.lmaxes[k][v] != actual_dcore.second[k][v]) {
					cout << edgesAdded << " has mismatching lmax["<<k<<"]["<<v<<"] " << m.lmaxes[k][v] << "!=" << actual_dcore.second[k][v] << endl;
					errors++;
					hasErrors = true;
				}
			}
		}
		// if (!hasErrors) continue;
		// for (degree k = 0; k < actual_dcore.size(); ++k) {
		// 	cout << "lmax["<<k<<"] = ";
		// 	for (vertex v = 0; v < m.V; ++v) {
		// 		cout << m.lmaxes[k][v] << " ";
		// 	}
		// 	cout << endl;
		// 	cout << "actu["<<k<<"] = ";
		// 	for (vertex v = 0; v < m.V; ++v) {
		// 		cout << actual_dcore[k][v] << " ";
		// 	}
		// 	cout << endl;
		// }
	}
	cout << "total errors: " << errors << endl;
#endif

	auto end = chrono::steady_clock::now();
	cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
}
