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

vector<degree> checkKmax(Graph &g, device_graph_pointers& g_p, device_accessory_pointers& a_p) {
	// move graph to GPU
	moveGraphToDevice(g, g_p);

	// ***** calculating k-max *****
	swapInOut(g_p); // do a flip!! (we're calculating the 0 l-list)
	auto [_, kmaxes] = KList(g, g_p, a_p, 0);
	swapInOut(g_p); // let's fix the mess we made...

	return kmaxes;
}

vector<vector<degree>> checkDCore(Graph &g, device_graph_pointers& g_p, device_accessory_pointers& a_p) {
	// move graph to GPU
	moveGraphToDevice(g, g_p);

	// ***** calculating k-max *****
	swapInOut(g_p); // do a flip!! (we're calculating the 0 l-list)
	auto [kmax, kmaxes] = KList(g, g_p, a_p, 0);
	swapInOut(g_p); // let's fix the mess we made...

	vector<vector<degree>> res;
	for (unsigned k = 0; k <= kmax; ++k) {
		refreshGraphOnGPU(g, g_p);	// degrees will be wrecked from previous calculation

		auto [_, core] = KList(g, g_p, a_p, k);
		res.push_back(core);
	}

	return res;
}





__global__ void kmaxCalculateED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) continue;

		// m_p.ED[v] = 0;

		offset start = g_p.in_neighbors_offset[v];
		offset end = g_p.in_neighbors_offset[v+1];
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
		if (v >= V) continue;

		m_p.PED[v] = m_p.ED[v];
		if (m_p.PED[v] == 0) continue;

		offset start = g_p.in_neighbors_offset[v];
		offset end = g_p.in_neighbors_offset[v+1];
		for (offset o = start; o < end; ++o) {
			vertex neighbor = g_p.in_neighbors[o];
			if (m_p.k_max[neighbor] == m_p.k_max[v] && m_p.ED[neighbor] <= m_p.k_max[v])
				--m_p.PED[v];
		}
	}
}

__global__ void kmaxFindUpperBound(device_maintenance_pointers m_p, unsigned V, vertex edgeFrom, vertex edgeTo) {
	__shared__ degree root_k_max;

	if (IS_MAIN_THREAD) {
		vertex root = edgeFrom;
		if (m_p.k_max[edgeTo] < m_p.k_max[edgeFrom])
			root = edgeTo;
		root_k_max = m_p.k_max[root];
	}

	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) continue;

		if (m_p.k_max[v] == root_k_max && m_p.PED[v] > m_p.k_max[v]) {
			m_p.compute[v] = true;
			m_p.k_max[v]++;
		}
	}
}

__global__ void kmaxRefineHIndex(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V) {
	__shared__ bool localFlag;

	if (IS_MAIN_THREAD)
		localFlag = false;
	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;

		if (v >= V) continue;
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
	if (IS_MAIN_THREAD)
		*m_p.flag = localFlag;
}

void kmaxMaintenance(Graph& g, device_graph_pointers& g_p, device_maintenance_pointers& m_p, pair<vertex, vertex> insertedEdge) {
	// cout << "\tLoading graph on GPU" << endl;
	// move graph to GPU
	moveGraphToDevice(g, g_p); // todo: this shouldnt be here for final maintenance

	// cout << "\tSetting up maintenance CUDA memory" << endl;
	initializeDeviceMaintenanceMemoryForKmax(g, m_p, g.kmaxes);

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
	kmaxFindUpperBound<<<BLOCK_NUMS, BLOCK_DIM>>>(m_p, g.V, insertedEdge.first, insertedEdge.second);

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
		kmaxRefineHIndex<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V);
		cudaMemcpy(&flag, m_p.flag, sizeof(bool), cudaMemcpyDeviceToHost);
	}

	// load back into cpu
	cudaMemcpy(g.kmaxes.data(), m_p.k_max, g.V * sizeof(degree), cudaMemcpyDeviceToHost);

	#ifdef PRINT_STEPS
	cout << "kmax_refine: ";
	for (auto v: g.kmaxes)
		cout << v << " ";
	cout << endl;
	#endif
}


__global__ void kListCalculateED(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) continue;

		// m_p.ED[v] = 0;
		if (m_p.k_max[v] < k) continue;

		offset start = g_p.out_neighbors_offset[v];
		offset end = g_p.out_neighbors_offset[v+1];
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
		if (v >= V) continue;

		m_p.PED[v] = m_p.ED[v];
		if (m_p.PED[v] == 0) continue;
		if (m_p.k_max[v] < k) continue;

		offset start = g_p.out_neighbors_offset[v];
		offset end = g_p.out_neighbors_offset[v+1];
		for (offset o = start; o < end; ++o) {
			vertex neighbor = g_p.out_neighbors[o];

			if (m_p.k_max[neighbor] < k) continue;

			if (m_p.l_max[neighbor] == m_p.l_max[v] && m_p.ED[neighbor] <= m_p.l_max[v])
				--m_p.PED[v];
		}
	}
}

__global__ void kListFindUpperBound(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k, vertex edgeFrom, vertex edgeTo) {
	__shared__ degree root_l_max;

	if (m_p.k_max[edgeFrom] < k || m_p.k_max[edgeTo] < k) return;

	if (IS_MAIN_THREAD) {
		vertex root = edgeFrom;
		if (m_p.l_max[edgeTo] < m_p.l_max[edgeFrom])
			root = edgeTo;
		root_l_max = m_p.l_max[root];
	}

	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;
		if (v >= V) continue;

		// if (m_p.k_max[v] >= k && m_p.l_max[v] <= root_l_max + 1 && m_p.PED[v] >= m_p.l_max[v]) {
		// if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max && m_p.PED[v] > m_p.l_max[v]) {
		// if (m_p.k_max[v] >= k && m_p.PED[v] > m_p.l_max[v]) {
		// if (m_p.k_max[v] >= k && m_p.l_max[v] == root_l_max) {
		if (m_p.k_max[v] >= k) {	// todo: this is pathetic... its rly just redoing the decomposition...
			m_p.compute[v] = true;
			// m_p.l_max[v] = g_p.out_degrees[v];
			// m_p.l_max[v] = k_adj_out[v].size();

			// // todo: this is probably rly expensive
			unsigned k_adj_out_v = 0;
			offset start = g_p.out_neighbors_offset[v];
			offset end = g_p.out_neighbors_offset[v+1];
			for (offset o = start; o < end; ++o) {
				vertex neighbor = g_p.out_neighbors[o];

				if (m_p.k_max[neighbor] < k) continue;

				k_adj_out_v++;
			}
			m_p.l_max[v] = k_adj_out_v;
		}
	}
}

__global__ void kListRefineHIndex(device_graph_pointers g_p, device_maintenance_pointers m_p, unsigned V, degree k) {
	__shared__ bool localFlag;

	if (IS_MAIN_THREAD)
		localFlag = false;
	__syncthreads();

	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < V; base += THREAD_COUNT) {
		vertex v = base + global_threadIdx;

		if (v >= V) continue;
		if (!m_p.compute[v]) continue;

		// if (m_p.k_max[v] < k) continue; // not necessary since this vertex would never be computed anyway

		// make histogram bucket (V+E) initialized to zero
		offset histogramStart = g_p.out_neighbors_offset[v] + v;
		offset histogramEnd = histogramStart + m_p.l_max[v];
		for (offset o = histogramStart; o <= histogramEnd; ++o)
			m_p.histograms[o] = 0;

		// put in neighbors into buckets
		offset inStart = g_p.in_neighbors_offset[v];
		offset inEnd = g_p.in_neighbors_offset[v+1];
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
		offset outEnd = g_p.out_neighbors_offset[v+1];
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
	if (IS_MAIN_THREAD)
		*m_p.flag = localFlag;
}

void kListMaintenance(Graph& g, device_graph_pointers& g_p, device_maintenance_pointers& m_p, pair<vertex, vertex> insertedEdge, degree k) {
	initializeDeviceMaintenanceMemoryForKList(g, m_p, g.lmaxes[k]);

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
	if (g.kmaxes[insertedEdge.first] >= k || g.kmaxes[insertedEdge.second] >= k) {
		vertex root = insertedEdge.first;
		if (g.lmaxes[k][insertedEdge.second] < g.lmaxes[k][insertedEdge.first])
			root = insertedEdge.second;
		cout << "root_l_max " << g.lmaxes[k][root] << endl;
	}
	#endif
	kListFindUpperBound<<<BLOCK_NUMS, BLOCK_DIM>>>(g_p, m_p, g.V, k, insertedEdge.first, insertedEdge.second);

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
	#endif


	unsigned count = 0;
	bool flag = true;
	while (flag) {
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

	cudaMemcpy(g.lmaxes[k].data(), m_p.l_max, g.V * sizeof(degree), cudaMemcpyDeviceToHost);
}


void maintenance(Graph& g, device_graph_pointers& g_p, device_maintenance_pointers& m_p, pair<vertex, vertex> insertedEdge) {
	moveGraphToDevice(g, g_p);

	// cout << "kmax" << endl;
	degree M = min(g.kmaxes[insertedEdge.first], g.kmaxes[insertedEdge.second]);
	kmaxMaintenance(g, g_p, m_p, insertedEdge);

	// we need to maintain the size of lmaxes to be the size of the biggest kmax
	degree max_kmax = *max_element(g.kmaxes.begin(), g.kmaxes.end());
	if (max_kmax+1 != g.lmaxes.size()) {
		unsigned prevSize = g.lmaxes.size();
		g.lmaxes.resize(max_kmax+1);

		for (unsigned i = prevSize; i < max_kmax+1; ++i)
			g.lmaxes[i] = vector<degree>(g.V);
	}

	if (M > max_kmax-1) M = max_kmax-1;
	for (degree k = 0; k <= M+1; ++k) {
		// cout << "lmax " << k << endl;
		kListMaintenance(g, g_p, m_p, insertedEdge, k);
	}
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

    /*remove empty buckets*/
    B.erase (std::remove_if (B.begin (), B.end (), [] (const auto& vv)
    {
        return vv.empty ();
    }), B.end ());

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

	// const string filename = "../dataset/wiki_vote";
	const string filename = "../dataset/congress";

	auto start = chrono::steady_clock::now();

    Graph g(filename);
    cout << "> " << filename  << " V: " << g.V << " E: " << g.E << endl;

	auto res = dcore(g);
	writeDCoreResultsText(res, "../results/cudares.txt", 16);


	// writeDCoreResults(res, "../results/cudares");
	// compareDCoreResults("../results/cudares", "../results/wiki_vote");
	// compareDCoreResults("../results/cudares", "../results/amazon0601");

	device_graph_pointers g_p;
	allocateDeviceGraphMemory(g, g_p);
	device_accessory_pointers a_p;
	allocateDeviceAccessoryMemory(g, a_p);
	device_maintenance_pointers m_p;
	allocateDeviceMaintenanceMemory(g, m_p);

	Graph m(g.V);
	unsigned edgesAdded = 0;
	unsigned errors = 0;
	for (auto edge: g.edges) {
		// if (edgesAdded == 22) break;
		m.insertEdge(edge);
		edgesAdded++;
		if (edgesAdded % 100 == 0)
			cout << edgesAdded << "/" << g.E << ": " << edge.first << "->" << edge.second << endl;

		bool hasErrors = false;

		maintenance(m, g_p, m_p, edge);
		auto actual_dcore = checkDCore(m, g_p, a_p);

		if (m.lmaxes.size() != actual_dcore.size()) {
			cout << edgesAdded << " has mismatching k-max sizes! " << m.lmaxes.size() << "!=" << actual_dcore.size() << endl;
			errors++;
			continue;
		}
		for (degree k = 0; k < actual_dcore.size(); ++k) {
			for (vertex v = 0; v < m.V; ++v) {
				if (m.lmaxes[k][v] != actual_dcore[k][v]) {
					cout << edgesAdded << " has mismatching lmax["<<k<<"]["<<v<<"] " << m.lmaxes[k][v] << "!=" << actual_dcore[k][v] << endl;
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


	// // insert edge by edge to create same decomp (K MAX MAINTENANCE)
	// Graph m(g.V);
	// unsigned edgesAdded = 0;
	// unsigned errors = 0;
	// for (auto edge: g.edges) {
	// 	m.insertEdge(edge);
	// 	edgesAdded++;
	// 	if (edgesAdded % 100 == 0)
	// 		cout << edgesAdded << "/" << g.E << endl;
	//
	// 	kmaxMaintenance(m, g_p, m_p, edge);
	// 	auto actual_kmax = checkKmax(m, g_p, a_p);
	//
	// 	// cout << "kmax_actua: ";
	// 	// for (auto v: actual_kmax)
	// 	// 	cout << v << " ";
	// 	// cout << endl;
	//
	// 	for (vertex v = 0; v < m.V; ++v) {
	// 		if (m.kmaxes[v] != actual_kmax[v]) {
	// 			cout << edgesAdded << " has mismatching kmax["<<v<<"] " << m.kmaxes[v] << "!=" << actual_kmax[v] << endl;
	// 			errors++;
	// 		}
	// 	}
	// }
	// cout << "total errors: " << errors << endl;


	// // cuda maintenance,,,
	// auto maintenanceStart = chrono::steady_clock::now();
	// kmaxMaintenance(g, {1, 0});
	// auto maintenanceEnd = chrono::steady_clock::now();
	// cout << "Maintenance time: " << chrono::duration_cast<chrono::milliseconds>(maintenanceEnd - maintenanceStart).count() << "ms" << endl;




	// auto kmaxBefore = g.kmaxes;
	// cout << "k-max-befr: ";
	// for (auto v: kmaxBefore)
	// 	cout << v << " ";
	// cout << endl;
	//
	// // lets try k-max maintenance
	// // vector<pair<vertex, vertex>> newEdges = {{1,0}, {6,0}, {6,3}};
	// vector<pair<vertex, vertex>> newEdges = {{6,2},{6,4},{6,5},{6,7},{2,6},{4,6},{5,6},{7,6}};
	// for (auto edge: newEdges) {
	// 	g.insertEdge(edge);
	// 	cout << "inserted edge " << edge.first << "->" << edge.second << endl;
	//
	// 	auto kmax = maintainKmax_(g, newEdges);
	//
	// 	cout << "k-max-calc: ";
	// 	for (auto v: kmax)
	// 		cout << v << " ";
	// 	cout << endl;
	//
	// 	auto check = checkKmax(g);
	// 	cout << "k-max-chck: ";
	// 	for (auto v: kmax)
	// 		cout << v << " ";
	// 	cout << endl;
	//
	// 	auto changed = 0;
	// 	for (int idx = 0; idx < kmaxBefore.size(); ++idx) {
	// 		if (kmaxBefore[idx] != kmax[idx]) {
	// 			changed++;
	// 		}
	// 	}
	// 	cout << changed << " values changed" << endl;
	//
	// 	changed = 0;
	// 	// cout << "checking...";
	// 	for (int idx = 0; idx < kmax.size(); ++idx) {
	// 		if (kmax[idx] != check[idx]) {
	// 			changed++;
	// 			cout << idx << ": " << kmax[idx] << " should be " << check[idx] << endl;
	// 		}
	// 	}
	// 	if (!changed)
	// 		cout << endl << "0 values mismatching" << endl;
	// 	if (changed)
	// 		cout << changed << " values mismatching" << endl;
	// }

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
