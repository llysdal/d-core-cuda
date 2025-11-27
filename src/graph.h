#ifndef CUDA_PLAYGROUND_GRAPH_H
#define CUDA_PLAYGROUND_GRAPH_H

#include "common.h"
#include <set>
#include <string>

using namespace std;


class Graph: public GraphInterface {
public:
	vector<vector<vertex>> inEdges;
	vector<vector<vertex>> outEdges;
	vector<pair<vertex, vertex>> edges;
	set<vertex> inserted;
	vertex* in_neighbors;
	vertex* out_neighbors;
	offset* in_neighbors_offset;
	offset* out_neighbors_offset;
	degree* in_degrees;
	degree* out_degrees;
	Graph();
	explicit Graph(const string& inputFile);
	explicit Graph(unsigned V);
	pair<vertex, vertex> getRandomInsertEdge() override;
	pair<vertex, vertex> getRandomDeleteEdge() override;
	void insertEdges(const vector<pair<vertex, vertex>>& edgesToBeInserted) override;
	void insertEdgesInPlace(const vector<pair<vertex, vertex>>& edgesToBeInserted) override;
	void deleteEdges(const vector<pair<vertex, vertex>>& edgesToBeDeleted) override;
	void deleteEdgesInPlace(const vector<pair<vertex, vertex>>& edgesToBeDeleted) override;
	void readFile(const string& inputFile);
	void writeFile(const string& outputFile);
	void writeBinary(const string& outputFile);
	bool readBinary(const string& inputFile);
	~Graph();
};

#endif //CUDA_PLAYGROUND_GRAPH_H

