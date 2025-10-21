#include "graph_gen.h"

#include <fstream>
#include <iostream>
#include <set>

#include "common.h"

void generateGraph(const string& outputFile, unsigned V, unsigned E) {
	srand(time(nullptr));

	set<pair<vertex, vertex>> edges;
	for (unsigned e = 0; e < E; e++) {
		pair<vertex, vertex> edge = {rand() % V, rand() % V};

		while (edge.first == edge.second || edges.contains(edge))
			edge = {rand() % V, rand() % V};
		edges.insert(edge);
	}

	ofstream outfile;
	outfile.open(outputFile);
	if (!outfile) {
		cout << "generating graph file "<<outputFile<<" failed " << endl;
	}

	outfile << "# " << V << "\n";
	for (auto edge: edges)
		outfile << edge.first << " " << edge.second << "\n";

	outfile.close();

	cout << "Generated graph file "<<outputFile<<"\n";
}
