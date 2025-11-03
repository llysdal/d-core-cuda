#ifndef CUDA_PLAYGROUND_GRAPH_H
#define CUDA_PLAYGROUND_GRAPH_H

#include "common.h"
#include <string>

using namespace std;


class Graph {
public:
	unsigned V;
	unsigned E;
	vector<vector<vertex>> inEdges;
	vector<vector<vertex>> outEdges;
	vector<pair<vertex, vertex>> edges;
	vertex* in_neighbors;
	vertex* out_neighbors;
	offset* in_neighbors_offset;
	offset* out_neighbors_offset;
	degree* in_degrees;
	degree* out_degrees;
	// unsigned int kmax, dmax;
	degree kmax;
	vector<degree> kmaxes;
	degree lmax;
	vector<vector<degree>> lmaxes;
	Graph();
	explicit Graph(const string& inputFile);
	explicit Graph(unsigned V);
	void insertEdges(const vector<pair<vertex, vertex>>& edgesToBeInserted);
	void insertEdgesInPlace(const vector<pair<vertex, vertex>>& edgesToBeInserted);
	void readFile(const string& inputFile);
	void writeBinary(const string& inputFile);
	bool readBinary(const string& inputFile);
	~Graph();
};

#endif //CUDA_PLAYGROUND_GRAPH_H

