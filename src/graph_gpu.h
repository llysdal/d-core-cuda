#ifndef D_CORE_CUDA_GRAPH_GPU_H
#define D_CORE_CUDA_GRAPH_GPU_H

#include <cassert>
#include <vector>
#include "graph.h"

using namespace std;

class GraphGPU: public GraphInterface {
public:
	device_graph_pointers g_p;
	GraphGPU();
	explicit GraphGPU(Graph& g, device_graph_pointers dgp);
	void insertEdges(const vector<pair<vertex, vertex>>& edges) override;
	void insertEdgesInPlace(const vector<pair<vertex, vertex>> &edgesToBeInserted) override {assert(false);};
	pair<vertex, vertex> getRandomInsertEdge() override {assert(false); return {0,0};};
	void deleteEdges(const vector<pair<vertex, vertex>>& edgesToBeDeleted) override {assert(false);};
	void deleteEdgesInPlace(const vector<pair<vertex, vertex>>& edgesToBeDeleted) override {assert(false);};
	pair<vertex, vertex> getRandomDeleteEdge() override {assert(false); return {0,0};};
	~GraphGPU();
};

#endif //D_CORE_CUDA_GRAPH_GPU_H