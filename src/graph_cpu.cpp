#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <sstream>
#include <set>
#include <numeric>

#include "./graph_cpu.h"



using namespace std;

GraphCPU::GraphCPU() {
	// default constructor
}

// GraphCPU& GraphCPU::operator=(const GraphCPU& g) {
// 	auto n = GraphCPU();
//
// 	n.V = g.V;
// 	n.E = g.E;
//
// 	n.inEdges = g.inEdges;
// 	n.outEdges = g.outEdges;
// 	n.inDegrees = vector<atomic<degree>>(V);
// 	n.outDegrees = vector<atomic<degree>>(V);
// 	for (int i=0; i<n.V; i++) {
// 		n.inDegrees[i] = n.inEdges[i].size();
// 		n.outDegrees[i] = n.outEdges[i].size();
// 	}
//
// 	return n;
// }

void GraphCPU::readFile(const string& inputFile) {
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
	edges = vector<pair<vertex, vertex>>();

	while (getline(infile, line)) {
		istringstream iss(line);
		iss >> s >> t;
		if (s == t) continue; // no self loops
		edges.emplace_back(s, t);
	}

	// done reading from the file
	infile.close();

	// amount of edges
	E = edges.size();

	// extract nodes with edges from node
	// nodes = vector<set<unsigned>>(V);
	inEdges = vector<vector<vertex>>(V);
	outEdges = vector<vector<vertex>>(V);
	for (auto &[first, second] : edges) {
		outEdges[first].push_back(second);
		inEdges[second].push_back(first);
	}

	// setup degrees
	inDegrees = vector<atomic<degree>>(V);
	outDegrees = vector<atomic<degree>>(V);
	for (int i=0; i<V; i++) {
		inDegrees[i] = inEdges[i].size();
		outDegrees[i] = outEdges[i].size();
	}

	// get average degree (this might overflow?)
	AVG_IN_DEGREE = accumulate(inDegrees.begin(), inDegrees.end(), 0.0f) / inDegrees.size();
	AVG_OUT_DEGREE = accumulate(outDegrees.begin(), outDegrees.end(), 0.0f) / outDegrees.size();


	// cout << edges[0].first << " " << edges[0].second << endl;
}

GraphCPU::GraphCPU(const string& inputFile){
    cout << "Graph reading file... " << endl;

    auto start = chrono::steady_clock::now();
    readFile(inputFile);
    auto end = chrono::steady_clock::now();

    cout << "Graph file loaded in: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
}
GraphCPU::~GraphCPU() {
    cout << "Graph deallocated... " << endl;
    // delete[] neighbors;
    // delete[] neighbors_offset;
    // delete[] degrees;
}