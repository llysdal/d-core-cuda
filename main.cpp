#include <algorithm>
#include <barrier>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ostream>
#include <ranges>
#include <semaphore>
#include <syncstream>
#include <thread>
#include <vector>

#include <omp.h>

#include "graph_cpu.h"

using namespace std;

// this destroys the graph btw
vector<degree> Klist(GraphCPU& D, unsigned k) {
	const unsigned n = D.V;

	int visited = 0;
	vector<bool> flag(D.V, true);	// this uses the total amount of vertices in the graph?
	int level = 0;

	int start = 0;
	int end = 0;
	vector<vertex> buf(n);

	while (visited < n) {

		// scan for out degree equal to current level or in degree less than k
		for (vertex v = 0; v < n; v++) {
			if (!flag[v]) continue;

			if (D.outDegrees[v] == level) {
				buf[end] = v; end++; flag[v] = false;
			}
			else if (D.inDegrees[v] < k) {
				buf[end] = v; end++; flag[v] = false;
				D.outDegrees[v] = level;	// should this be atomic?
			}
		}


		while (start < end) {
			vertex v = buf[start]; start++; flag[v] = false;

			for (vertex u: D.outEdges[v]) {
				if (!flag[u]) continue;

				degree d = (D.inDegrees[u]--);	// atomic
				if (d <= k) {
					buf[end] = u; end++; flag[u] = false;
					D.outDegrees[u] = level;
				}
			}

			for (vertex w: D.inEdges[v]) {
				if (!flag[w]) continue;

				if (D.outDegrees[w] > level) {
					degree d = (D.outDegrees[w]--);	// atomic
					if (d == (level + 1)) {
						buf[end] = w; end++; flag[w] = false;
					}
					else if (d <= level) {
						++D.outDegrees[w]; flag[w] = false;
					}
				}
			}
		}

		visited += end;	// atomic
		start = 0;
		end = 0;
		level++;
	}

	vector<degree> res(D.V);
	for (int i = 0; i < D.V; ++i) {
		res[i] = D.inDegrees[i];
	}
	return res;
}


vector<degree> PKlist(GraphCPU& D, unsigned k, unsigned threads) {
	const unsigned n = D.V;

	atomic<int> visited = 0;
	vector<bool> flag(D.V, true);	// this uses the total amount of vertices in the graph?

	// we need prints!
	osyncstream bout(cout);

	{
		vector<jthread> t;
		barrier sync(threads);
		unsigned chunkSize = ceil((float)n/(float)threads);

		for (int thread = 0; thread < threads; ++thread) {
			auto startIdx = thread * chunkSize;
			auto endIdx = min((thread+1) * chunkSize - 1, n - 1);

			t.emplace_back([&, startIdx, endIdx]() {
				int start = 0;
				int end = 0;
				int level = 0;
				vector<vertex> buf(n);

				while (visited < n) {
					// scan for out degree equal to current level or in degree less than k
					for (vertex v = startIdx; v <= endIdx; v++) {
						if (!flag[v]) continue;

						if (D.outDegrees[v] == level) {
							buf[end] = v; end++; flag[v] = false;
						}
						else if (D.inDegrees[v] < k) {
							buf[end] = v; end++; flag[v] = false;
							D.outDegrees[v] = level;	// should this be atomic?
						}
					}

					while (start < end) {
						vertex v = buf[start]; start++; flag[v] = false;

						for (vertex u: D.outEdges[v]) {
							if (!flag[u]) continue;

							degree d = (D.inDegrees[u]--);	// atomic
							if (d <= k) {
								buf[end] = u; end++; flag[u] = false;
								D.outDegrees[u] = level;		// atomic..?
							}
						}

						for (vertex w: D.inEdges[v]) {
							if (!flag[w]) continue;

							if (D.outDegrees[w] > level) {
								degree d = (D.outDegrees[w]--);	// atomic
								if (d == level + 1) {
									buf[end] = w; end++; flag[w] = false;
								}
								else if (d <= level) {
									++D.outDegrees[w]; flag[w] = false;
								}
							}
						}
					}

					visited += end;
					start = 0;
					end = 0;
					level++;

					sync.arrive_and_wait();
				}
			});
		};
	}

	vector<degree> res(D.V);
	for (int i = 0; i < D.V; ++i) {
		res[i] = D.inDegrees[i];
	}
	return res;
}

void Decompose(string filename, unsigned kmax, unsigned threads) {
	vector<vector<degree>> res;

	for (int k = 0; k <= kmax; ++k) {
		GraphCPU g(filename);
		res.push_back(PKlist(g, k, threads));
	}

	ofstream outfile ("./parres.txt",ios::in|ios::out|ios::binary|ios::trunc);
	for(int i = 0; i < res.size(); i++) {
		for(int j = 0; j < res[i].size(); j++){
			outfile << res[i][j] << " ";
		}
		outfile << "\r\n";
	}
}



int main(int argc, char *argv[]) {
	const string filename = "../dataset/congress.txt";

    // cout << "Graph loading started... " << endl;
    // GraphCPU g(filename);
	// GraphCPU g2(filename);
 //    cout << ">" << filename << endl;
 //    cout << "V: " << g.V << endl;
 //    cout << "E: " << g.E << endl;
	// cout << "AVG_IN_DEGREE: " << g.AVG_IN_DEGREE << endl;
	// cout << "AVG_OUT_DEGREE: " << g.AVG_OUT_DEGREE << endl;
	// cout << "AVG_DEGREE: " << (g.E * 2.0f) / g.V << endl;

	Decompose(filename, 32, 1);

	// auto start = chrono::steady_clock::now();
	// auto a = PKlist(g, 0, 1);
	// auto end = chrono::steady_clock::now();
	// cout << "we're done, it took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
	//
	// start = chrono::steady_clock::now();
	// auto b = PKlist(g2, 0, 2);
	// end = chrono::steady_clock::now();
	// cout << "we're done, it took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
	//
	// unsigned errors = 0;
	// for (int i = 0; i < a.size(); ++i) {
	// 	if (a[i] != b[i])
	// 		++errors;
	// }
	// cout << "errors: " << errors << endl;

	cout << "we're done" << endl;
}
