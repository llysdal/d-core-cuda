#include <fstream>
#include <iostream>
#include <chrono>
#include <sstream>
#include <numeric>
#include <vector>

#include "./graph.h"


using namespace std;

Graph::Graph() {
	// default constructor
}


void Graph::readFile(const string& inputFile) {
    ifstream infile;
    infile.open(inputFile);
    if (!infile) {
        cout << "loading graph file failed " << endl;
    }

	string line;

	// first line is vertex amount
	getline(infile, line);
	V = stoi(line.substr(2, line.length() - 2));

	// all the next are edges
	vertex s,t;
	auto edges = vector<pair<vertex, vertex>>();

	while (getline(infile, line)) {
		istringstream iss(line);
		iss >> s >> t;
		if (s == t) continue; // no self loops
		edges.emplace_back(s, t);
	}

	// done reading from the file
	infile.close();
	cout << "> Read in data..." << endl;
	// amount of edges
	E = edges.size();

	// extract nodes with edges from node
	auto inEdges = vector<vector<vertex>>(V);
	auto outEdges = vector<vector<vertex>>(V);
	for (auto &[first, second] : edges) {
		outEdges[first].push_back(second);
		inEdges[second].push_back(first);
	}

	cout << "> Extracted edges..." << endl;

	// setup degrees
	in_degrees = new degree[V];
	out_degrees = new degree[V];
	for (int i=0; i<V; i++) {
		in_degrees[i] = inEdges[i].size();
		out_degrees[i] = outEdges[i].size();
	}
	cout << "> Set up degrees..." << endl;

	in_neighbors_offset = new offset[V+1];
	in_neighbors_offset[0] = 0;
	partial_sum(in_degrees, in_degrees+V, in_neighbors_offset+1);
	out_neighbors_offset = new offset[V+1];
	out_neighbors_offset[0] = 0;
	partial_sum(out_degrees, out_degrees+V, out_neighbors_offset+1);

	cout << "> Set up offsets..." << endl;

	E_IN = in_neighbors_offset[V];
	E_OUT = out_neighbors_offset[V];

	in_neighbors = new vertex[E_IN];
	out_neighbors = new vertex[E_OUT];

	#pragma omp parallel for
	for (vertex v = 0; v < V; v++) {
		auto it = inEdges[v].begin();
		for (offset j = in_neighbors_offset[v]; j < in_neighbors_offset[v+1]; j++, it++)
			in_neighbors[j] = *it;
	}
	#pragma omp parallel for
	for (vertex v = 0; v < V; v++) {
		auto it = outEdges[v].begin();
		for (offset j = out_neighbors_offset[v]; j < out_neighbors_offset[v+1]; j++, it++)
			out_neighbors[j] = *it;
	}

	cout << "> Set up neighbors..." << endl;

	// get average degree (this might overflow?)
	// AVG_IN_DEGREE = accumulate(inDegrees.begin(), inDegrees.end(), 0.0f) / inDegrees.size();
	// AVG_OUT_DEGREE = accumulate(outDegrees.begin(), outDegrees.end(), 0.0f) / outDegrees.size();
}

Graph::Graph(const string& inputFile){
    cout << "Graph reading file... " << endl;

    auto start = chrono::steady_clock::now();
    readFile(inputFile);
    auto end = chrono::steady_clock::now();

    cout << "Graph file loaded in: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
}
Graph::~Graph() {
    delete[] in_neighbors;
    delete[] out_neighbors;
    delete[] in_neighbors_offset;
    delete[] out_neighbors_offset;
    delete[] in_degrees;
    delete[] out_degrees;
    cout << "Graph deallocated... " << endl;
}