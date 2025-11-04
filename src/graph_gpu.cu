#include "common.h"
#include "graph_gpu.h"

#include <numeric>

GraphGPU::GraphGPU() {

}

GraphGPU::GraphGPU(Graph& g, device_graph_pointers dgp) {
	V = g.V;
	E = 0;

	kmaxes = vector<degree>(V);
	lmaxes = vector<vector<degree>>();
	// for (int i = 0; i < g.kmax; i++)
		lmaxes.emplace_back(vector<degree>(V));

	g_p = dgp;
	cudaMemset(g_p.in_degrees, 0, g.V * sizeof(vertex));
	cudaMemset(g_p.out_degrees, 0, g.V * sizeof(vertex));

	// offset* neighbors_offset = new offset[V+1];
	// neighbors_offset[0] = 0;
	// for (int i = 0; i < V+1; i++)
	// 	neighbors_offset[i] = i * OFFSET_GAP;
	// cudaMemcpy(g_p.in_neighbors_offset, neighbors_offset, (g.V + 1) * sizeof(offset), cudaMemcpyHostToDevice);
	// cudaMemcpy(g_p.out_neighbors_offset, neighbors_offset, (g.V + 1) * sizeof(offset), cudaMemcpyHostToDevice);


	cudaMemcpy(g_p.in_neighbors_offset, g.in_neighbors_offset, (g.V + 1) * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.out_neighbors_offset, g.out_neighbors_offset, (g.V + 1) * sizeof(offset), cudaMemcpyHostToDevice);
}

__global__ void graphInsertEdges(device_graph_pointers g_p, unsigned edgeAmount) {
	unsigned global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned base = 0; base < edgeAmount; base += THREAD_COUNT) {
		unsigned e = base + global_threadIdx;
		if (e >= edgeAmount) break;

		vertex edgeFrom = g_p.modified_edges[e*2];
		vertex edgeTo = g_p.modified_edges[e*2+1];

		offset indeg = atomicAdd(g_p.in_degrees + edgeTo, 1);
		g_p.in_neighbors[g_p.in_neighbors_offset[edgeTo] + indeg] = edgeFrom;

		offset outdeg = atomicAdd(g_p.out_degrees + edgeFrom, 1);
		g_p.out_neighbors[g_p.out_neighbors_offset[edgeFrom] + outdeg] = edgeTo;
	}
}

void GraphGPU::insertEdges(const vector<pair<vertex, vertex>>& edges) {
	E += edges.size();
	graphInsertEdges<<<BLOCK_COUNT, BLOCK_DIM>>>(g_p, edges.size());
}

GraphGPU::~GraphGPU() {

}