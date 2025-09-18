#ifndef CUDA_PLAYGROUND_GRAPH_H
#define CUDA_PLAYGROUND_GRAPH_H

// #include "./common.h"
#include <atomic>
#include <string>
#include <set>
#include <vector>

using namespace std;

typedef unsigned vertex;
typedef unsigned degree;

class GraphCPU {
	public:
	unsigned int V;
	unsigned int E;
	double AVG_IN_DEGREE = 0;
	double AVG_OUT_DEGREE = 0;
	vector<vector<vertex>> inEdges;
	vector<vector<vertex>> outEdges;
	vector<pair<vertex, vertex>> edges;
	// std::vector<unsigned int> neighbors;
	// std::vector<unsigned int> neighbors_offset;
	vector<degree> inDegrees;
	vector<degree> outDegrees;
	// unsigned int kmax, dmax;
	GraphCPU();
	GraphCPU& operator=(const GraphCPU&);
	explicit GraphCPU(const string& inputFile);
	void readFile(const string& inputFile);
	~GraphCPU();
};

#endif //CUDA_PLAYGROUND_GRAPH_H

