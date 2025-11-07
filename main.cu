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
		cudaMemset(a_p.bufferTails, 0, BLOCK_COUNT * sizeof(unsigned));

		scan<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, a_p, k, level, g.V);
		process<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, a_p, k, level);

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
	findMaxKmax<<<BLOCK_COUNT, BLOCK_DIM>>>(m_p, V);
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
	findMValue<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, modifiedEdges.size());
	degree M = 0;
	cudaMemcpy(&M, m_p.m_value, sizeof(degree), cudaMemcpyDeviceToHost);
	return M;
}

#ifdef PED_NOWARP
__global__ void kmaxCalculateED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + GLOBAL_THREAD_ID;
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
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + GLOBAL_THREAD_ID;
		if (v >= V) break;

		m_p.PED[v] = m_p.ED[v];
		if (m_p.PED[v] == 0) continue;

		offset start = g_p.in_neighbors_offset[v];
		offset end = start + g_p.in_degrees[v];
		for (offset o = start; o < end; ++o) {
			vertex neighbor = g_p.in_neighbors[o];
			if (m_p.k_max[neighbor] == m_p.k_max[v] && m_p.ED[neighbor] <= m_p.k_max[v])
				--m_p.PED[v];
		}
	}
}
#endif

#ifdef PED_WARP
__global__ void kmaxCalculateED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	for (unsigned base = 0; base < V; base += WARP_COUNT) {
		vertex v = base + GLOBAL_WARP_ID;
		if (v >= V) break;

		// m_p.ED[v] = 0;

		offset start = g_p.in_neighbors_offset[v];
		offset end = start + g_p.in_degrees[v];
		for (offset o = start; o < end; o += WARP_SIZE) {
			if (o + LANE_ID >= end) break;

			vertex neighbor = g_p.in_neighbors[o + LANE_ID];
			if (m_p.k_max[neighbor] >= m_p.k_max[v])
				atomicAdd(m_p.ED + v, 1);
		}
	}
}

__global__ void kmaxCalculatePED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	for (unsigned base = 0; base < V; base += WARP_COUNT) {
		vertex v = base + GLOBAL_WARP_ID;
		if (v >= V) break;

		if (IS_MAIN_IN_WARP)
			m_p.PED[v] = m_p.ED[v];
		__syncwarp();
		if (m_p.PED[v] == 0) continue;

		offset start = g_p.in_neighbors_offset[v];
		offset end = start + g_p.in_degrees[v];
		for (offset o = start; o < end; o += WARP_SIZE) {
			if (o + LANE_ID >= end) break;

			vertex neighbor = g_p.in_neighbors[o + LANE_ID];
			if (m_p.k_max[neighbor] == m_p.k_max[v] && m_p.ED[neighbor] <= m_p.k_max[v])
				atomicSub(m_p.PED + v, 1);
		}
	}
}
#endif

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

	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + GLOBAL_THREAD_ID;
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
	for (unsigned edgebase = 0; edgebase < edges; edgebase += BLOCK_COUNT) {
		unsigned e = edgebase + BLOCK_ID;
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
			vertex v = base + THREAD_ID;
			if (v >= V) break;

			if (m_p.k_max[v] == root_k_max && m_p.PED[v] > m_p.k_max[v]) {
				if (atomicTestAndSet(&m_p.compute[v]))
					m_p.k_max[v]++;
			}
		}
	}
}

#ifdef HINDEX_NOWARP
__global__ void kmaxRefineHIndex(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	__shared__ bool localFlag;

	if (IS_MAIN_THREAD) localFlag = false;
	__syncthreads();

	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + GLOBAL_THREAD_ID;
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
#endif

#ifdef HINDEX_WARP
__global__ void kmaxRefineHIndex(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	__shared__ bool localFlag;

	if (IS_MAIN_THREAD) localFlag = false;
	__syncthreads();

	for (unsigned base = 0; base < V; base += WARP_COUNT) {
		vertex v = base + GLOBAL_WARP_ID;
		if (v >= V) break;

		if (!m_p.compute[v]) continue;

		offset histogramStart = g_p.in_neighbors_offset[v] + v;
		offset histogramEnd = histogramStart + m_p.k_max[v];
		for (offset o = histogramStart; o <= histogramEnd; o += WARP_SIZE)
			if (o + LANE_ID <= histogramEnd)
				m_p.histograms[o + LANE_ID] = 0;
		__syncwarp();

		offset start = g_p.in_neighbors_offset[v];
		offset end = start + g_p.in_degrees[v];
		for (offset o = start; o < end; o += WARP_SIZE) {
			if (o + LANE_ID >= end) continue;
			vertex neighbor = g_p.in_neighbors[o + LANE_ID];

			atomicAdd(m_p.histograms + (histogramStart + min(m_p.k_max[v],m_p.k_max[neighbor])), 1);
		}
		__syncwarp();

		// calculate h index
		degree tmp_h_index = 0;
		if (IS_MAIN_IN_WARP)
			tmp_h_index = hOutIndex(m_p, v, g_p.in_neighbors_offset[v], m_p.k_max[v]);
		__syncwarp();

		if (IS_MAIN_IN_WARP) {
			if (tmp_h_index < m_p.k_max[v]) {
				m_p.k_max[v] = tmp_h_index;
				localFlag = true;
			}
		}
	}

	__syncthreads();
	if (IS_MAIN_THREAD && localFlag)
		*m_p.flag = true;
}
#endif

// we expect modified edges to already be set in device graph pointers!!
// we also expect kmax on gpu to be "in place" as in, we wont load it from graphdata or place it into graphdata
void kmaxMaintenance(unsigned V, device_graph_pointers& g_p, device_maintenance_pointers& m_p, vector<pair<vertex, vertex>>& modifiedEdges) {
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
	initializeDeviceMaintenanceMemory(V, m_p);

	// cout << "\tCalculating ED and PED" << endl;
	kmaxCalculateED<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V);
	kmaxCalculatePED<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V);
	#ifdef PRINT_STEPS
	cout << "             ";
	for (vertex v = 0; v < V; ++v)
		cout << v << " ";
	cout << endl;
	vector<degree> ED(V);
	vector<degree> PED(V);
	cudaMemcpy(ED.data(), m_p.ED, V * sizeof(degree), cudaMemcpyDeviceToHost);
	cudaMemcpy(PED.data(), m_p.PED, V * sizeof(degree), cudaMemcpyDeviceToHost);

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
		kmaxFindUpperBoundBatch<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V, modifiedEdges.size());
	else
		kmaxFindUpperBound<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V);

	#ifdef PRINT_STEPS
	vector<degree> upper(V);
	cudaMemcpy(upper.data(), m_p.k_max, V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "   kmax_upp: ";
	for (auto v: upper)
		cout << v << " ";
	cout << endl;
	vector<unsigned> compute(V);
	cudaMemcpy(compute.data(), m_p.compute, V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "compute    : ";
	for (auto v: compute)
		cout << v << " ";
	cout << endl;
	#endif


	bool flag = true;
	while (flag) {
		cudaMemset(m_p.flag, false, sizeof(bool));
		kmaxRefineHIndex<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V);
		cudaMemcpy(&flag, m_p.flag, sizeof(bool), cudaMemcpyDeviceToHost);
	}

	// load back into cpu
	// getKmaxFromDeviceMemory(m_p, g.kmaxes);

	#ifdef PRINT_STEPS
	vector<degree> kmax_refine(V);
	cudaMemcpy(kmax_refine.data(), m_p.k_max, V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "kmax_refine: ";
	for (auto v: kmax_refine)
		cout << v << " ";
	cout << endl << endl;
	#endif
}

__global__ void kmaxFindUpperBoundDelete(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
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

	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + GLOBAL_THREAD_ID;
		if (v >= V) break;

		if (m_p.k_max[v] == root_k_max && m_p.ED[v] <= m_p.k_max[v]) {
			m_p.compute[v] = true;
			if (m_p.k_max[v] > 0) m_p.k_max[v]++;
		}
	}
}

void kmaxMaintenanceDelete(unsigned V, device_graph_pointers& g_p, device_maintenance_pointers& m_p, vector<pair<vertex, vertex>>& modifiedEdges) {
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
	initializeDeviceMaintenanceMemory(V, m_p);

	// cout << "\tCalculating ED" << endl;
	kmaxCalculateED<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V);

	#ifdef PRINT_STEPS
	cout << "             ";
	for (vertex v = 0; v < V; ++v)
		cout << v << " ";
	cout << endl;
	vector<degree> ED(V);
	cudaMemcpy(ED.data(), m_p.ED, V * sizeof(degree), cudaMemcpyDeviceToHost);

	cout << "         ED: ";
	for (auto v: ED)
		cout << v << " ";
	cout << endl;
	#endif

	// cout << "\tCalculating upper bounds" << endl;
	// we would load modified edges here if they werent already loaded
	assert(modifiedEdges.size() == 1); // we only do single edge maintenance for now
		// kmaxFindUpperBoundBatch<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V, modifiedEdges.size());
	kmaxFindUpperBoundDelete<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V);

	#ifdef PRINT_STEPS
	vector<degree> upper(V);
	cudaMemcpy(upper.data(), m_p.k_max, V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "   kmax_upp: ";
	for (auto v: upper)
		cout << v << " ";
	cout << endl;
	vector<unsigned> compute(V);
	cudaMemcpy(compute.data(), m_p.compute, V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "compute    : ";
	for (auto v: compute)
		cout << v << " ";
	cout << endl;
	#endif

	bool flag = true;
	while (flag) {
		cudaMemset(m_p.flag, false, sizeof(bool));
		kmaxRefineHIndex<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V);
		cudaMemcpy(&flag, m_p.flag, sizeof(bool), cudaMemcpyDeviceToHost);
	}

	#ifdef PRINT_STEPS
	vector<degree> kmax_refine(V);
	cudaMemcpy(kmax_refine.data(), m_p.k_max, V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "kmax_refine: ";
	for (auto v: kmax_refine)
		cout << v << " ";
	cout << endl << endl;
	#endif
}


#ifdef PED_NOWARP
__global__ void kListCalculateED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + GLOBAL_THREAD_ID;
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
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + GLOBAL_THREAD_ID;
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
// __global__ void kListCalculatePED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
// 	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
// 		vertex v = base + GLOBAL_THREAD_ID;
// 		if (v >= V) break;
//
// 		m_p.PED[v] = 0;
// 		if (m_p.k_max[v] < k) continue;
//
// 		offset start = g_p.out_neighbors_offset[v];
// 		offset end = start + g_p.out_degrees[v];
// 		for (offset o = start; o < end; ++o) {
// 			vertex neighbor = g_p.out_neighbors[o];
//
// 			if (m_p.k_max[neighbor] < k) continue;
//
// 			if (m_p.l_max[neighbor] == m_p.l_max[v] && m_p.ED[neighbor] > m_p.l_max[v])
// 				++m_p.PED[v];
// 			if (m_p.l_max[neighbor] > m_p.l_max[v])
// 				++m_p.PED[v];
// 		}
// 	}
// }
#endif

#ifdef PED_WARP
__global__ void kListCalculateED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	for (unsigned base = 0; base < V; base += WARP_COUNT) {
		vertex v = base + GLOBAL_WARP_ID;
		if (v >= V) break;

		// m_p.ED[v] = 0;
		if (m_p.k_max[v] < k) continue;

		offset start = g_p.out_neighbors_offset[v];
		offset end = start + g_p.out_degrees[v];
		for (offset o = start; o < end; o += WARP_SIZE) {
			if (o + LANE_ID >= end) break;

			vertex neighbor = g_p.out_neighbors[o + LANE_ID];

			if (m_p.k_max[neighbor] < k) continue;

			// we need to maintain some sort of k_adj_in/out list here?

			if (m_p.l_max[neighbor] >= m_p.l_max[v])
				atomicAdd(m_p.ED + v, 1);
		}
	}

}

__global__ void kListCalculatePED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	for (unsigned base = 0; base < V; base += WARP_COUNT) {
		vertex v = base + GLOBAL_WARP_ID;
		if (v >= V) break;

		m_p.PED[v] = m_p.ED[v];
		if (m_p.PED[v] == 0) continue;
		if (m_p.k_max[v] < k) continue;

		offset start = g_p.out_neighbors_offset[v];
		offset end = start + g_p.out_degrees[v];
		for (offset o = start; o < end; o += WARP_SIZE) {
			if (o + LANE_ID >= end) break;

			vertex neighbor = g_p.out_neighbors[o + LANE_ID];

			if (m_p.k_max[neighbor] < k) continue;

			if (m_p.l_max[neighbor] == m_p.l_max[v] && m_p.ED[neighbor] <= m_p.l_max[v])
				atomicSub(m_p.PED + v, 1);
		}
	}
}
#endif

__global__ void kListFindUpperBound(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	__shared__ degree root_l_max;

	if (IS_MAIN_THREAD) {
		vertex edgeFrom = g_p.modified_edges[0];
		vertex edgeTo = g_p.modified_edges[1];

		vertex root = edgeFrom;
		if (m_p.l_max[edgeTo] < m_p.l_max[edgeFrom])
			root = edgeTo;
		root_l_max = m_p.l_max[root];
	}

	__syncthreads();

	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + GLOBAL_THREAD_ID;
		if (v >= V) break;

		// if (m_p.k_max[v] >= k && m_p.l_max[v] <= root_l_max + 1 && m_p.PED[v] >= m_p.l_max[v]) {
		#ifdef USE_RESTRICTIVE_KLIST_COMPUTE_MASK
		if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max && m_p.PED[v] > m_p.l_max[v]) {
		#endif
		// if (m_p.k_max[v] >= k && m_p.PED[v] > m_p.l_max[v]) {
		// if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max) {
		#ifndef USE_RESTRICTIVE_KLIST_COMPUTE_MASK
		if (m_p.k_max[v] >= k) {	// todo: this is pathetic... its rly just redoing the decomposition...
		#endif
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
	__shared__ degree root_l_max;

	// each block works on an edge at a time :)
	for (unsigned edgebase = 0; edgebase < edges; edgebase += BLOCK_COUNT) {
		unsigned e = edgebase + BLOCK_ID;
		if (e >= edges) break;

		if (IS_MAIN_THREAD) {
			vertex edgeFrom = g_p.modified_edges[e*2];
			vertex edgeTo = g_p.modified_edges[e*2+1];

			skip_edge = (m_p.k_max[edgeFrom] < k || m_p.k_max[edgeTo] < k);	// todo: is this right for unbatched lmax maint?
			// skip_edge = false;

			vertex root = edgeFrom;
			if (m_p.l_max[edgeTo] < m_p.l_max[edgeFrom])
				root = edgeTo;
			root_l_max = m_p.l_max[root];
		}

		__syncthreads();
		if (skip_edge) continue;

		for (unsigned base = 0; base < V; base += BLOCK_DIM) {
			vertex v = base + THREAD_ID;
			if (v >= V) break;

			// if (m_p.k_max[v] >= k && m_p.l_max[v] <= root_l_max + 1 && m_p.PED[v] >= m_p.l_max[v]) {
			#ifdef USE_RESTRICTIVE_KLIST_COMPUTE_MASK
			if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max && m_p.PED[v] > m_p.l_max[v]) {
			#endif
			// if (m_p.k_max[v] >= k && m_p.PED[v] > m_p.l_max[v]) {
			// if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max) {
			#ifndef USE_RESTRICTIVE_KLIST_COMPUTE_MASK
			if (m_p.k_max[v] >= k) {	// todo: this is pathetic... its rly just redoing the decomposition...
			#endif
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

	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + GLOBAL_THREAD_ID;

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
		vertex v = base + GLOBAL_WARP_ID;
		if (v >= V) break;

		if (!m_p.compute[v]) continue;

		// if (m_p.k_max[v] < k) continue; // not necessary since this vertex would never be computed anyway

		// make histogram bucket (V+E) initialized to zero
		offset histogramStart = g_p.out_neighbors_offset[v] + v;
		offset histogramEnd = histogramStart + m_p.l_max[v];
		for (offset o = histogramStart; o <= histogramEnd; o += WARP_SIZE)
			if (o + LANE_ID <= histogramEnd)
				m_p.histograms[o + LANE_ID] = 0;
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
		__syncwarp();

		offset outStart = g_p.out_neighbors_offset[v];
		offset outEnd = outStart + g_p.out_degrees[v];
		for (offset o = outStart; o < outEnd; o += WARP_SIZE) {
			if (o + LANE_ID >= outEnd) continue;
			vertex neighbor = g_p.out_neighbors[o + LANE_ID];

			if (m_p.k_max[neighbor] < k) continue;

			atomicAdd(m_p.histograms + (histogramStart + min(m_p.l_max[v],m_p.l_max[neighbor])), 1);
		}
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

void kListMaintenance(unsigned V, device_graph_pointers& g_p, device_maintenance_pointers& m_p, vector<pair<vertex, vertex>>& modifiedEdges, degree k) {
	#ifdef PRINT_STEPS
		cout << "L MAX for k = " << k << endl;;
	#endif

	initializeDeviceMaintenanceMemory(V, m_p);

	kListCalculateED<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V, k);
	kListCalculatePED<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V, k);
	#ifdef PRINT_STEPS
	cout << "             ";
	for (vertex v = 0; v < V; ++v)
		cout << v << " ";
	cout << endl;
	vector<degree> ED(V);
	vector<degree> PED(V);
	cudaMemcpy(ED.data(), m_p.ED, V * sizeof(degree), cudaMemcpyDeviceToHost);
	cudaMemcpy(PED.data(), m_p.PED, V * sizeof(degree), cudaMemcpyDeviceToHost);

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
		kListFindUpperBoundBatch<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V, modifiedEdges.size(), k);
	else
		kListFindUpperBound<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V, k);

	#ifdef PRINT_STEPS
	vector<degree> upper(V);
	cudaMemcpy(upper.data(), m_p.l_max, V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "lmax["<<k<<"]_upp: ";
	for (auto v: upper)
		cout << v << " ";
	cout << endl;
	vector<unsigned> compute(V);
	cudaMemcpy(compute.data(), m_p.compute, V * sizeof(unsigned), cudaMemcpyDeviceToHost);
	cout << "compute    : ";
	for (auto v: compute)
		cout << v << " ";
	cout << endl;
	unsigned count = 0;
	#endif

	bool flag = true;
	while (flag) {
		cudaMemset(m_p.flag, false, sizeof(bool));

		cudaMemcpy(m_p.new_l_max, m_p.l_max, V * sizeof(degree), cudaMemcpyDeviceToDevice);
		kListRefineHIndex<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V, k);
		cudaMemcpy(m_p.l_max, m_p.new_l_max, V * sizeof(degree), cudaMemcpyDeviceToDevice);

		cudaMemcpy(&flag, m_p.flag, sizeof(bool), cudaMemcpyDeviceToHost);

		#ifdef PRINT_STEPS
		cout << "round_cnt: " << count++ << endl;
		vector<degree> new_l_max(V);
		cudaMemcpy(new_l_max.data(), m_p.l_max, V * sizeof(degree), cudaMemcpyDeviceToHost);
		cout << "  lmax["<<k<<"] =  ";
		for (vertex v = 0; v < V; ++v)
			cout << new_l_max[v] << " ";
		cout << endl;
		#endif
	}

	#ifdef PRINT_STEPS
	cout << endl;
	#endif
}

__global__ void kListFindUpperBoundDelete(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	__shared__ degree root_l_max;

	if (IS_MAIN_THREAD) {
		vertex edgeFrom = g_p.modified_edges[0];
		vertex edgeTo = g_p.modified_edges[1];

		vertex root = edgeFrom;
		if (m_p.l_max[edgeTo] < m_p.l_max[edgeFrom])
			root = edgeTo;
		root_l_max = m_p.l_max[root];
	}

	__syncthreads();

	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + GLOBAL_THREAD_ID;
		if (v >= V) break;

		// this removes lmax values that shouldnt be there due to reduced kmax
		if (m_p.k_max[v] < k && m_p.l_max[v] > 0)
			m_p.l_max[v] = 0;

		#ifdef USE_RESTRICTIVE_KLIST_COMPUTE_MASK
		if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max && m_p.ED[v] <= m_p.l_max[v]) {
		#endif
		#ifndef USE_RESTRICTIVE_KLIST_COMPUTE_MASK
		if (m_p.k_max[v] >= k) {	// todo: this is pathetic... its rly just redoing the decomposition...
		#endif
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

void kListMaintenanceDelete(unsigned V, device_graph_pointers& g_p, device_maintenance_pointers& m_p, vector<pair<vertex, vertex>>& modifiedEdges, degree k) {
	#ifdef PRINT_STEPS
		cout << "L MAX for k = " << k << endl;;
	#endif

	initializeDeviceMaintenanceMemory(V, m_p);

	kListCalculateED<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V, k);
	#ifdef PRINT_STEPS
	cout << "             ";
	for (vertex v = 0; v < V; ++v)
		cout << v << " ";
	cout << endl;
	vector<degree> ED(V);
	cudaMemcpy(ED.data(), m_p.ED, V * sizeof(degree), cudaMemcpyDeviceToHost);

	cout << "         ED: ";
	for (auto v: ED)
		cout << v << " ";
	cout << endl;
	#endif

	assert(modifiedEdges.size() == 1); // we only do single edge maintenance for now
	kListFindUpperBoundDelete<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V, k);

	#ifdef PRINT_STEPS
	vector<degree> upper(V);
	cudaMemcpy(upper.data(), m_p.l_max, V * sizeof(degree), cudaMemcpyDeviceToHost);
	cout << "lmax["<<k<<"]_upp: ";
	for (auto v: upper)
		cout << v << " ";
	cout << endl;
	vector<unsigned> compute(V);
	cudaMemcpy(compute.data(), m_p.compute, V * sizeof(unsigned), cudaMemcpyDeviceToHost);
	cout << "compute    : ";
	for (auto v: compute)
		cout << v << " ";
	cout << endl;
	unsigned count = 0;
	#endif

	bool flag = true;
	while (flag) {
		cudaMemset(m_p.flag, false, sizeof(bool));

		cudaMemcpy(m_p.new_l_max, m_p.l_max, V * sizeof(degree), cudaMemcpyDeviceToDevice);
		kListRefineHIndex<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, m_p, V, k);
		cudaMemcpy(m_p.l_max, m_p.new_l_max, V * sizeof(degree), cudaMemcpyDeviceToDevice);

		cudaMemcpy(&flag, m_p.flag, sizeof(bool), cudaMemcpyDeviceToHost);

		#ifdef PRINT_STEPS
		cout << "round_cnt: " << count++ << endl;
		vector<degree> new_l_max(V);
		cudaMemcpy(new_l_max.data(), m_p.l_max, V * sizeof(degree), cudaMemcpyDeviceToHost);
		cout << "  lmax["<<k<<"] =  ";
		for (vertex v = 0; v < V; ++v)
			cout << new_l_max[v] << " ";
		cout << endl;
		#endif
	}

	#ifdef PRINT_STEPS
	cout << endl;
	#endif
}


vector<vector<pair<vertex, vertex>>> getEdgeBatches(vector<degree> kmaxes, const vector<pair<vertex, vertex>>& edgesToBeInserted) {
	vector<vector<pair<vertex, vertex>>> batches;

	degree maxKmax = 0;
	vector<degree> edgeRoot(edgesToBeInserted.size());
	vector<degree> edgeKmax(edgesToBeInserted.size());
	for (unsigned eid = 0; eid < edgesToBeInserted.size(); ++eid) {
		auto edge = edgesToBeInserted[eid];
		vertex root = edge.second;
		if (kmaxes[edge.first] > kmaxes[edge.second])
			root = edge.first;
		edgeRoot[eid] = root;
		edgeKmax[eid] = kmaxes[root];
		maxKmax = max(maxKmax, kmaxes[root]);
	}
	// buckets
	vector<vector<unsigned>> B(maxKmax + 1);
	for (unsigned eid = 0; eid < edgesToBeInserted.size(); ++eid) {
		B[edgeKmax[eid]].push_back(eid);
	}

	for (auto& batch: B) { // each batch is a list of edges with the same kmax value
		if (batch.empty()) continue;

		bool flag = true;
		while (flag) {
			vector<pair<vertex, vertex>> candidateBatch;
			vector<bool> v_ (kmaxes.size(), false);

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

void maintenance(GraphInterface& g, device_graph_pointers& g_p, device_maintenance_pointers& m_p, vector<pair<vertex, vertex>>& modifiedEdges, bool isDelete = false, bool inPlace = false) {
	chrono::duration<double, milli> insertTime{}, kmaxTime{}, klistTime{};

	auto batches = getEdgeBatches(g.kmaxes, modifiedEdges);
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

		if (Graph* _ = dynamic_cast<Graph*>(&g)) {
			insertTime += funcTime([&] {
				putModifiedEdgesInDeviceMemory(g_p, batch);
				if (inPlace) {
					if (!isDelete) g.insertEdgesInPlace(batch);
					else g.deleteEdgesInPlace(batch);
				} else {
					assert(FORCE_REBUILD_GRAPH);
					if (!isDelete) g.insertEdges(batch);
					else g.deleteEdges(batch);
				}
				moveGraphToDevice(static_cast<Graph&>(g), g_p);
			});
			kmaxTime += funcTime([&] {
				putKmaxInDeviceMemory(m_p, g.kmaxes);
				M = max(M, getMValue(g_p, m_p,batch));
				if (!isDelete) kmaxMaintenance(g.V, g_p, m_p, batch);
				else kmaxMaintenanceDelete(g.V, g_p, m_p, batch);
				getKmaxFromDeviceMemory(m_p, g.kmaxes);
			});
		} else if (GraphGPU* _ = dynamic_cast<GraphGPU*>(&g)) {
			insertTime += funcTime([&] {
				putModifiedEdgesInDeviceMemory(g_p, batch);
				g.insertEdges(batch);
				// moveGraphToDevice(g, g_p);
			});
			kmaxTime += funcTime([&] {
				M = max(M, getMValue(g_p, m_p,batch));
				kmaxMaintenance(g.V, g_p, m_p, batch);
				getKmaxFromDeviceMemory(m_p, g.kmaxes); // needed because edge batches uses it...
			});
		}
	}

	getMaxKmax(g.kmax, g.V, m_p);

	if (g.kmax+1 != g.lmaxes.size()) {
		unsigned prevSize = g.lmaxes.size();
		g.lmaxes.resize(g.kmax+1);

		for (unsigned i = prevSize; i < g.kmax+1; ++i)
			g.lmaxes[i] = vector<degree>(g.V);
	}

	putModifiedEdgesInDeviceMemory(g_p, modifiedEdges);
	if (M > g.kmax-1) M = g.kmax-1;
	#ifdef PRINT_MAINTENANCE_STATS
	cout << "M+1=" << M+1 << " ";
	#endif
	klistTime = funcTime([&] {
		for (degree k = 0; k <= M+1; ++k) {
			putLmaxInDeviceMemory(m_p, g.lmaxes[k]);
			if (!isDelete) kListMaintenance(g.V, g_p, m_p, modifiedEdges, k);
			else kListMaintenanceDelete(g.V, g_p, m_p, modifiedEdges, k);
			getLmaxFromDeviceMemory(m_p, g.lmaxes[k]);
		}
	});

	#ifdef PRINT_MAINTENANCE_STATS
	cout << (isDelete ? "delete: " : "insert: ") << insertTime << " kmax: " << kmaxTime << " klist: " << klistTime << endl;
	#endif
}


int main(int argc, char *argv[]) {
	// generateGraph("../dataset/random", 10, 40);

	// const string filename = "../dataset/digraph";
	// const string filename = "../dataset/example2";
	// const string filename = "../dataset/wiki_vote";
	// const string filename = "../dataset/wiki_vote-scramble"; //2.5 times speedup, due to better batching
	// const string filename = "../dataset/email";
	// const string filename = "../dataset/email-scramble";
	// const string filename = "../dataset/live_journal";
	const string filename = "../dataset/hollywood"; // 2156s load time lol, 554s decomp time, more than 50gb ram used

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

// #define SINGLE_INSERT
#define SINGLE_DELETE
// #define SPEED_TEST
// #define STEP_TEST

	// ***********************************************r*******************************
#ifdef SINGLE_INSERT
	assert(OFFSET_GAP >= 1);	// gapsize needs to be atleast 1;
	// vector<pair<vertex, vertex>> edgeToAdd = {{0, 1}};	// digraph
	// vector<pair<vertex, vertex>> edgeToAdd = {{1, 2}};	// example2
	// vector<pair<vertex, vertex>> edgeToAdd = {{456, 316}};	// wiki_vote
	// vector<pair<vertex, vertex>> edgeToAdd = {{1, 0}};	// email
	// vector<pair<vertex, vertex>> edgeToAdd = {{50, 23}};	// email
	// vector<pair<vertex, vertex>> edgeToAdd = {{47, 46}};	// email
	vector<pair<vertex, vertex>> edgeToAdd = {{2, 1}};	// live_journal
	// vector<pair<vertex, vertex>> edgeToAdd = {};

	auto maintStart = chrono::steady_clock::now();
	maintenance(g, g_p, m_p, edgeToAdd, false, true);
	cout << "Maintenance total time \t\t" << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - maintStart).count() << "ms" << endl;

	if (!SINGlE_INSERT_SKIP_CHECK) {
		auto comp = checkDCore(g, g_p, a_p);
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
#ifdef SINGLE_DELETE
		// vector<pair<vertex, vertex>> edgeToDelete = {{7, 5}};	// digraph
		// vector<pair<vertex, vertex>> edgeToDelete = {{3, 30}};	// wiki_vote
		// vector<pair<vertex, vertex>> edgeToDelete = {{1, 44}};	// email
		// vector<pair<vertex, vertex>> edgeToDelete = {{0, 1}};	// hollywood (M+1=7)
		// vector<pair<vertex, vertex>> edgeToDelete = {{13, 12}};	// hollywood (M+1=38)
		vector<pair<vertex, vertex>> edgeToDelete = {{323, 343}};// hollywood (M+1=60)
		// vector<pair<vertex, vertex>> edgeToDelete = {};

		auto maintStart = chrono::steady_clock::now();
		maintenance(g, g_p, m_p, edgeToDelete, true, true);
		cout << "Maintenance total time \t\t" << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - maintStart).count() << "ms" << endl;

		if (!SINGlE_INSERT_SKIP_CHECK) {
			auto comp = checkDCore(g, g_p, a_p);
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
	unsigned batchSize = 1000;
	assert(MODIFIED_EDGES_BUFFER_SIZE >= batchSize*2);	// Ensure all the edges will fit in the buffer
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
	unsigned batchSize = 1;
	assert(MODIFIED_EDGES_BUFFER_SIZE >= batchSize*2);	// Ensure all the edges will fit in the buffer
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

		maintenance(m, g_p, m_p, edgesToAdd);
		auto actual_dcore = checkDCore(m, g_p, a_p);

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
				}
			}
		}
	}
	cout << "total errors: " << errors << endl;
#endif

	auto end = chrono::steady_clock::now();
	cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
}
