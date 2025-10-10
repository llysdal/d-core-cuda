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

void Graph::insertEdge(pair<vertex, vertex> edge) {
	edges.emplace_back(edge.first, edge.second);

	E = edges.size();

	// todo: optimize
	auto inEdges = vector<vector<vertex>>(V);
	auto outEdges = vector<vector<vertex>>(V);
	for (auto &[first, second] : edges) {
		outEdges[first].push_back(second);
		inEdges[second].push_back(first);
	}

	in_degrees = new degree[V];
	out_degrees = new degree[V];
	for (int i=0; i<V; i++) {
		in_degrees[i] = inEdges[i].size();
		out_degrees[i] = outEdges[i].size();
	}

	in_neighbors_offset = new offset[V+1];
	in_neighbors_offset[0] = 0;
	partial_sum(in_degrees, in_degrees+V, in_neighbors_offset+1);
	out_neighbors_offset = new offset[V+1];
	out_neighbors_offset[0] = 0;
	partial_sum(out_degrees, out_degrees+V, out_neighbors_offset+1);

	in_neighbors = new vertex[E];
	out_neighbors = new vertex[E];

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
	edges = vector<pair<vertex, vertex>>();

	while (getline(infile, line)) {
		istringstream iss(line);
		iss >> s >> t;
		if (s == t) continue; // no self loops
		edges.emplace_back(s, t);
	}

	// done reading from the file
	infile.close();
	cout << "> Read in data..." << endl;

	// todo: we oughta trim the empty vertices away from the set here

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

	in_neighbors = new vertex[E];
	out_neighbors = new vertex[E];

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
}

void Graph::writeBinary(const string& inputFile) {
	ofstream bin;
	bin.open(inputFile + string("-binary"), ios::binary | ios::out);

	if (bin) {
		cout << "Graph writing binary..." << endl;
		auto start = chrono::steady_clock::now();
		bin.write(reinterpret_cast<char*>(&V), sizeof(unsigned));
		bin.write(reinterpret_cast<char*>(&E), sizeof(unsigned));
		bin.write(reinterpret_cast<char*>(in_degrees), static_cast<streamsize>(V * sizeof(degree)));
		bin.write(reinterpret_cast<char*>(out_degrees), static_cast<streamsize>(V * sizeof(degree)));
		bin.write(reinterpret_cast<char*>(in_neighbors_offset), static_cast<streamsize>((V + 1) * sizeof(offset)));
		bin.write(reinterpret_cast<char*>(out_neighbors_offset), static_cast<streamsize>((V + 1) * sizeof(offset)));
		bin.write(reinterpret_cast<char*>(in_neighbors), static_cast<streamsize>(E * sizeof(vertex)));
		bin.write(reinterpret_cast<char*>(out_neighbors), static_cast<streamsize>(E * sizeof(vertex)));

		bin.close();
		auto end = chrono::steady_clock::now();
		cout << "Graph binary written\t\t" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
	} else {
		cout << inputFile + string("-binary") << ": could not open file" << endl;
	}
}

bool Graph::readBinary(const string& inputFile) {
	ifstream bin;
	bin.open(inputFile + string("-binary"), ios::binary | ios::in);
	if (bin) {
		cout << "Graph reading binary..." << endl;
		auto start = chrono::steady_clock::now();
		bin.read(reinterpret_cast<char*>(&V), sizeof(unsigned));
		bin.read(reinterpret_cast<char*>(&E), sizeof(unsigned));
		in_degrees = new degree[V];
		out_degrees = new degree[V];
		in_neighbors_offset = new offset[V+1];
		out_neighbors_offset = new offset[V+1];
		in_neighbors = new vertex[E];
		out_neighbors = new vertex[E];
		bin.read(reinterpret_cast<char*>(in_degrees), static_cast<streamsize>(V * sizeof(degree)));
		bin.read(reinterpret_cast<char*>(out_degrees), static_cast<streamsize>(V * sizeof(degree)));
		bin.read(reinterpret_cast<char*>(in_neighbors_offset), static_cast<streamsize>((V + 1) * sizeof(offset)));
		bin.read(reinterpret_cast<char*>(out_neighbors_offset), static_cast<streamsize>((V + 1) * sizeof(offset)));
		bin.read(reinterpret_cast<char*>(in_neighbors), static_cast<streamsize>(E * sizeof(vertex)));
		bin.read(reinterpret_cast<char*>(out_neighbors), static_cast<streamsize>(E * sizeof(vertex)));

		bin.close();
		auto end = chrono::steady_clock::now();
		cout << "Graph binary read\t\t" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
		return true;
	} else {
		cout << inputFile + string("-binary") << ": could not open file" << endl;
	}
	return false;
}

Graph::Graph(const string& inputFile){
	// if (readBinary(inputFile)) return;

    cout << "Graph reading file..." << endl;
    auto start = chrono::steady_clock::now();
    readFile(inputFile);
    auto end = chrono::steady_clock::now();
    cout << "Graph file loaded\t\t" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

	// writeBinary(inputFile);
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