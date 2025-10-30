#ifndef D_CORE_CUDA_GRAPH_GPU_H
#define D_CORE_CUDA_GRAPH_GPU_H

#include <vector>
#include "graph.h"

using namespace std;

class GraphGPU {
public:
	unsigned V;
	unsigned E;
	device_graph_pointers g_p;
	degree kmax;
	vector<degree> kmaxes;
	degree lmax;
	vector<vector<degree>> lmaxes;
	GraphGPU();
	explicit GraphGPU(Graph& g, device_graph_pointers dgp);
	void insertEdges(const vector<pair<vertex, vertex>>& edges);
	~GraphGPU();
};

#endif //D_CORE_CUDA_GRAPH_GPU_H