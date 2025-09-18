#include <algorithm>
// #include <barrier>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
// #include <ranges>
// #include <semaphore>
// #include <syncstream>
#include <thread>
#include <vector>

#include <omp.h>

#include "graph_cpu.h"

using namespace std;

// this destroys the graph btw
// vector<degree> Klist(GraphCPU& D, unsigned k) {
// 	const unsigned n = D.V;
//
// 	int visited = 0;
// 	vector<bool> flag(D.V, true);	// this uses the total amount of vertices in the graph?
// 	int level = 0;
//
// 	int start = 0;
// 	int end = 0;
// 	vector<vertex> buf(n);
//
// 	while (visited < n) {
//
// 		// scan for out degree equal to current level or in degree less than k
// 		for (vertex v = 0; v < n; v++) {
// 			if (!flag[v]) continue;
//
// 			if (D.outDegrees[v] == level) {
// 				buf[end] = v; end++; flag[v] = false;
// 			}
// 			else if (D.inDegrees[v] < k) {
// 				D.outDegrees[v] = level;	// should this be atomic?
// 				buf[end] = v; end++; flag[v] = false;
// 			}
// 		}
//
//
// 		while (start < end) {
// 			vertex v = buf[start]; start++; flag[v] = false;
//
// 			for (vertex u: D.outEdges[v]) {
// 				if (!flag[u]) continue;
//
// 				degree d = (D.inDegrees[u]--);	// atomic
// 				if (d <= k) {
// 					buf[end] = u; end++; flag[u] = false;
// 					D.outDegrees[u] = level;
// 				}
// 			}
//
// 			for (vertex w: D.inEdges[v]) {
// 				if (!flag[w]) continue;
//
// 				if (D.outDegrees[w] > level) {
// 					degree d = (D.outDegrees[w]--);	// atomic
// 					if (d == (level + 1)) {
// 						buf[end] = w; end++; flag[w] = false;
// 					}
// 					else if (d <= level) {
// 						++D.outDegrees[w]; flag[w] = false;
// 					}
// 				}
// 			}
// 		}
//
// 		visited += end;	// atomic
// 		start = 0;
// 		end = 0;
// 		level++;
// 	}
//
// 	vector<degree> res(D.V);
// 	for (int i = 0; i < D.V; ++i) {
// 		res[i] = D.inDegrees[i];
// 	}
// 	return res;
// }
//
//
// vector<degree> PKlist(GraphCPU& D, unsigned k, unsigned threads) {
// 	const unsigned n = D.V;
//
// 	atomic<int> visited = 0;
// 	vector<bool> flag(D.V, true);	// this uses the total amount of vertices in the graph?
//
// 	// we need prints!
// 	osyncstream bout(cout);
//
// 	{
// 		vector<jthread> t;
// 		barrier sync(threads);
// 		unsigned chunkSize = ceil((float)n/(float)threads);
//
// 		for (int thread = 0; thread < threads; ++thread) {
// 			auto startIdx = thread * chunkSize;
// 			auto endIdx = min((thread+1) * chunkSize - 1, n - 1);
//
// 			t.emplace_back([&, startIdx, endIdx]() {
// 				int start = 0;
// 				int end = 0;
// 				int level = 0;
// 				vector<vertex> buf(n);
//
// 				while (visited < n) {
// 					// scan for out degree equal to current level or in degree less than k
// 					for (vertex v = startIdx; v <= endIdx; v++) {
// 						if (!flag[v]) continue;
//
// 						if (D.outDegrees[v] == level) {
// 							buf[end] = v; end++; flag[v] = false;
// 						}
// 						else if (D.inDegrees[v] < k) {
// 							D.outDegrees[v] = level;	// should this be atomic?
// 							buf[end] = v; end++; flag[v] = false;
// 						}
// 					}
//
// 					while (start < end) {
// 						vertex v = buf[start]; start++; flag[v] = false;
//
// 						for (vertex u: D.outEdges[v]) {
// 							if (!flag[u]) continue;
//
// 							degree d = (D.inDegrees[u]--);	// atomic
// 							if (d <= k) {
// 								if (!flag[u]) continue;
// 								buf[end] = u; end++; flag[u] = false;
// 								D.outDegrees[u] = level;		// atomic..?
// 							}
// 						}
//
// 						for (vertex w: D.inEdges[v]) {
// 							if (!flag[w]) continue;
//
// 							if (D.outDegrees[w] > level) {
// 								degree d = (D.outDegrees[w]--);	// atomic
// 								if (d == level + 1) {
// 									buf[end] = w; end++; flag[w] = false;
// 								}
// 								else if (d <= level) {
// 									++D.outDegrees[w]; flag[w] = false;
// 								}
// 							}
// 						}
// 					}
//
// 					visited += end;
// 					start = 0;
// 					end = 0;
// 					level++;
//
// 					sync.arrive_and_wait();
// 				}
// 			});
// 		};
// 	}
//
// 	vector<degree> res(D.V);
// 	for (int i = 0; i < D.V; ++i) {
// 		res[i] = D.inDegrees[i];
// 	}
// 	return res;
// }
//
// void Decompose(string filename, unsigned kmax, unsigned threads) {
// 	vector<vector<degree>> res;
//
// 	for (int k = 0; k <= kmax; ++k) {
// 		GraphCPU g(filename);
// 		res.push_back(Klist(g, k));
// 	}
//
// 	ofstream outfile ("./parres.txt",ios::in|ios::out|ios::binary|ios::trunc);
// 	for(int i = 0; i < res.size(); i++) {
// 		for(int j = 0; j < res[i].size(); j++){
// 			outfile << setw(3);
// 			outfile << res[i][j] << " ";
// 		}
// 		outfile << "\r\n";
// 	}
// }


std::vector<int> Parpeel_org(GraphCPU& g, int k){
	unsigned n = g.V;
    int NUM_THREADS = omp_get_max_threads();

    int visited = 0;


    std::vector<int> Din, Dout, flag;
    Din.resize(n);
    Dout.resize(n);
    flag.resize(n);
    for (int i = 0; i < n; i++) {
    	Dout[i] = g.outDegrees[i];
    	Din[i] = g.inDegrees[i];
        flag[i] = 0;
    }

#pragma omp parallel
{
    int level = 0;

    long buff_size = n;

    int *buff = (int *)malloc(buff_size*sizeof(int));
    assert(buff != NULL);

    int start = 0, end = 0;

    while(visited < n){
        #pragma omp for schedule(static)
        for(int i = 0; i < n; i++){
            if(flag[i] < 1){
                if(Dout[i] == level){
                    buff[end] = i;
                    end++;
                    flag[i]++;
                }
                else if (Din[i] < k){
                    Dout[i] = level;
                    buff[end] = i;
                    end++;
                    flag[i]++;
                }
            }
        }


        while(start < end){
            int v = buff[start];
            start++;
            flag[v]++;

            for(int j = 0; j < g.outEdges[v].size(); j++){
                int u = g.outEdges[v][j];

                if(flag[u]) continue;

                int din_u = __sync_fetch_and_sub(&Din[u], 1);

                if(din_u <= k){
                    if(flag[u] == 0){
                        buff[end] = u;
                        end++;

                        Dout[u] = level;
                        flag[u]++;
                    }
                }

            }

            for(int j = 0; j < g.inEdges[v].size(); j++){
                int u = g.inEdges[v][j];

                if(flag[u]) continue;

                int dout_u = Dout[u];

                if(dout_u > level){
                    int du = __sync_fetch_and_sub(&Dout[u], 1);
                    if(du==(level+1)){
                        buff[end] = u;
                        end++;
                        flag[u]++;
                    }

                    if(du <= level){
                        __sync_fetch_and_add(&Dout[u], 1);
                        flag[u]++;
                    }
                }

            }
        }

        __sync_fetch_and_add(&visited, end);

        #pragma omp barrier
        start = 0;
        end = 0;
        level = level+1;
    }

    free( buff );
}

    return Dout;
}


void PDC_org(GraphCPU& g){
	int num_of_thread = 8;
	unsigned n = g.V;


    omp_set_num_threads(num_of_thread);
    int NUM_THREADS = num_of_thread;


    int vis_num = 0;

    std::vector<int> Din, Dout;
    Din.resize(n);
    Dout.resize(n);
    for (int i = 0; i < n; i++) {
        Dout[i] = g.outDegrees[i];
        Din[i] = g.inDegrees[i];
    }

    auto begin = std::chrono::steady_clock::now();




#pragma omp parallel
{
    int level = 0;

    long buff_size = n;


    int *buff = (int *)malloc(buff_size*sizeof(int));
    assert(buff != NULL);

    int start = 0, end = 0;

    while(vis_num < n){

        #pragma omp for schedule(static)
        for(int i = 0; i < n; i++){
            if(Din[i] == level){
                buff[end] = i;
                end++;
            }
        }

        while(start < end){
            int v = buff[start];
            start++;

            for(int j = 0; j < g.outEdges[v].size(); j++){
                int u = g.outEdges[v][j];
                int din_u = Din[u];

                if(din_u > level){
                    int du = __sync_fetch_and_sub(&Din[u], 1);
                    if(du==(level+1)){
                        buff[end] = u;
                        end++;
                    }

                    if(du <= level) __sync_fetch_and_add(&Din[u], 1);
                }

            }
        }

        __sync_fetch_and_add(&vis_num, end);

        #pragma omp barrier
        start = 0;
        end = 0;
        level = level+1;
    }

    free(buff);
}

    int level = *max_element(Din.begin(),Din.end());


    std::vector<std::vector<int>> Fres;

    Fres.push_back(Parpeel_org(g, 0));

    auto end1 = std::chrono::steady_clock::now();

    for (int k=1; k<=level;k++) {
        Fres.push_back(Parpeel_org(g, k));
    }

    auto end2 = std::chrono::steady_clock::now();

	std::ofstream outfile ("./res.txt",ios::in|ios::out|ios::binary|ios::trunc);
	for (int i = 0; i < Fres.size(); i++) {
		for(int j = 0; j < Fres[i].size(); j++){
			outfile << setw(3);
			if (Fres[i][j] > 20) //this should be lmax?
				outfile << -1 << " ";
			else
				outfile << Fres[i][j] << " ";
		}
		outfile << "\r\n";
	}

    double runtime1 = std::chrono::duration<double>(end1 - begin).count();
    double runtime2 = std::chrono::duration<double>(end2 - end1).count();
    printf("stage 1 running time: %.4f sec, stage 2 running time: %.4lf sec.\n", runtime1, runtime2);

}

int main(int argc, char *argv[]) {
	const string filename = "../dataset/congress.txt";

    // cout << "Graph loading started... " << endl;
    GraphCPU g(filename);
	// GraphCPU g2(filename);
	cout << ">" << filename << endl;
	cout << "V: " << g.V << endl;
	cout << "E: " << g.E << endl;
	// cout << "AVG_IN_DEGREE: " << g.AVG_IN_DEGREE << endl;
	// cout << "AVG_OUT_DEGREE: " << g.AVG_OUT_DEGREE << endl;
	// cout << "AVG_DEGREE: " << (g.E * 2.0f) / g.V << endl;

	// Decompose(filename, 15, 1);

	PDC_org(g);

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
