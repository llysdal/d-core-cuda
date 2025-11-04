#ifndef CUDA_PLAYGROUND_GRAPH_H
#define CUDA_PLAYGROUND_GRAPH_H

#include "common.h"
#include <string>

using namespace std;


class Graph: public GraphInterface {
public:
	vector<vector<vertex>> inEdges;
	vector<vector<vertex>> outEdges;
	vector<pair<vertex, vertex>> edges;
	vertex* in_neighbors;
	vertex* out_neighbors;
	offset* in_neighbors_offset;
	offset* out_neighbors_offset;
	degree* in_degrees;
	degree* out_degrees;
	Graph();
	explicit Graph(const string& inputFile);
	explicit Graph(unsigned V);
	void insertEdges(const vector<pair<vertex, vertex>>& edgesToBeInserted) override;
	void insertEdgesInPlace(const vector<pair<vertex, vertex>>& edgesToBeInserted) override;
	void readFile(const string& inputFile);
	void writeBinary(const string& inputFile);
	bool readBinary(const string& inputFile);
	~Graph();
};

#endif //CUDA_PLAYGROUND_GRAPH_H

