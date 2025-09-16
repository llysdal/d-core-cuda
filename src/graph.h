#ifndef CUDA_PLAYGROUND_GRAPH_H
#define CUDA_PLAYGROUND_GRAPH_H

#include "./common.h"
#include <string>

using namespace std;

typedef unsigned vertex;
typedef unsigned offset;
typedef unsigned degree;

class Graph {
public:
	unsigned V;
	unsigned E;
	unsigned E_IN;
	unsigned E_OUT;
	double AVG_IN_DEGREE = 0;
	double AVG_OUT_DEGREE = 0;
	vertex* in_neighbors;
	vertex* out_neighbors;
	offset* in_neighbors_offset;
	offset* out_neighbors_offset;
	degree* in_degrees;
	degree* out_degrees;
	// unsigned int kmax, dmax;
	Graph();
	explicit Graph(const string& inputFile);
	void readFile(const string& inputFile);
	~Graph();
};

#endif //CUDA_PLAYGROUND_GRAPH_H

