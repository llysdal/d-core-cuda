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

int num_of_thread = 8;

//parallel peeling for a given k, advanced version
vector<int> Parpeel(GraphCPU& g, int k, vector<int> kBin, int* vert){
	unsigned n = g.V;

    //获取线程数
    int NUM_THREADS = num_of_thread;
    // omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel
    {
        #pragma omp master
        NUM_THREADS = omp_get_num_threads();
    }

    int visited = 0;

    // printf("number of threads: %d, k: %d.\n", NUM_THREADS,k);


    //重新写一个缩图
    std::vector<int> iH, OC, flag, shellVert;
    iH.resize(n);
    OC.resize(n);
    flag.resize(n,0);

    for (int i = 0; i < n; i++) {
        OC[i] = g.outDegrees[i];
        iH[i] = g.inDegrees[i];
    }

    //去除shell小于k的顶点,不在shell中的out-core number为0
    for(int i=kBin[k];i<n;i++)
        shellVert.push_back(vert[i]);

    for(int i = 0;i<kBin[k];i++){
        flag[vert[i]] = 1;
        OC[vert[i]] = 0;
    }

    //更新shellVert中顶点的degree
    for(int i=kBin[k];i<n;i++){
        int v = vert[i];
        for (int l = 0; l < g.outEdges[v].size(); l++) {
            if (flag[g.outEdges[v][l]]) OC[v]--;
        }
        for (int k = 0; k < g.inEdges[v].size(); k++) {
            if (flag[g.inEdges[v][k]]) iH[v]--;
        }
    }

    int n1 = shellVert.size();


    //每个线程运行这部分代码
#pragma omp parallel
{
    int level = 0;

    long buff_size = n;

    int *buff = (int *)malloc(buff_size*sizeof(int));
    assert(buff != NULL);

    int start = 0, end = 0;

    while(visited < n1){
        //获取out-degree为level的顶点, 或in-degree 小于k的顶点
        #pragma omp for schedule(static)
        for(int i = 0; i < n1; i++){
            int u = shellVert[i];
            if(flag[u] < 1){
                if(OC[u] == level){
                    buff[end] = u;
                    end++;
                    flag[u]++;
                }
                else if (iH[u] < k){
                    OC[u] = level;
                    buff[end] = u;
                    end++;
                    flag[u]++;
                }
            }
        }

        //处理buff中的顶点, 需要给小于din小于k的顶点加一个flag, 首先初始化所有顶点的flag为0,如果小于k, 那就将其修改为1, 并将其放入buff中
        while(start < end){
            int v = buff[start];
            start++;
            flag[v]++;

            //处理in-degree小于k的顶点
            for(int j = 0; j < g.outEdges[v].size(); j++){
                int u = g.outEdges[v][j];

                if(flag[u]) continue;

                // int din_u = __sync_fetch_and_sub(&iH[u], 1);
                int din_u = __sync_fetch_and_sub(&iH[u], 1);

                //in-degree小于k的顶点, 其l-number为当前的level
                if(din_u <= k){
                    if(flag[u] == 0){
                        buff[end] = u;
                        end++;

                        OC[u] = level;
                        flag[u]++;
                    }
                }

            }

            //处理out-degree不大于level的顶点
            for(int j = 0; j < g.inEdges[v].size(); j++){
                int u = g.inEdges[v][j];

                if(flag[u]) continue;

                int dout_u = OC[u];

                if(dout_u > level){
                    int du = __sync_fetch_and_sub(&OC[u], 1);
                    //将减小后degree等于level的顶点放在buff末尾
                    if(du==(level+1)){
                        buff[end] = u;
                        end++;
                        flag[u]++;
                    }

                    if(du <= level){
                        __sync_fetch_and_add(&OC[u], 1);
                        flag[u]++;
                    }
                }

            }

            // printf("start: %d, end: %d.\n",start,end);
        }

        //减去处理过的顶点数
        __sync_fetch_and_add(&visited, end);


        // printf("end: %d, n1: %d.\n",end, n1);
        // printf("level: %d, visited: %d.\n",level, visited);

        #pragma omp barrier
        start = 0;
        end = 0;
        level = level+1;

    }

    free( buff );
}

    return OC;
}

//parallel peeling for a given k, advanced version
vector<int> Parpeel(GraphCPU& g, int k, vector<int> upper, vector<int> kBin, int* vert) {
	unsigned int n = g.V;
    int NUM_THREADS = omp_get_max_threads();

    #pragma omp parallel
    {
        #pragma omp master
        NUM_THREADS = omp_get_num_threads();
    }

    int visited = 0;



    std::vector<int> Din, Dout, flag, shellVert, core;
    Din.resize(n);
    Dout.resize(n);
    flag.resize(n,0);
    core.resize(n);

    for (int i = 0; i < n; i++) {
        Dout[i] = g.outDegrees[i];
        Din[i] = g.inDegrees[i];
        core[i] = -1;
    }

    for(int i=kBin[k];i<n;i++)
        shellVert.push_back(vert[i]);

    int lmax = 0;
    for(int i = 0;i<kBin[k];i++){
        flag[vert[i]] = 1;
        Dout[vert[i]] = 0;
        if(upper[vert[i]]>lmax) lmax = upper[vert[i]];
    }

    for(int i=kBin[k];i<n;i++){
        int v = vert[i];
        for (int l = 0; l < g.outEdges[v].size(); l++) {
            if (flag[g.outEdges[v][l]]) Dout[v]--;
        }
        for (int k = 0; k < g.inEdges[v].size(); k++) {
            if (flag[g.inEdges[v][k]]) Din[v]--;
        }
    }

    int n1 = shellVert.size();


#pragma omp parallel
{
    int level = 0;

    long buff_size = n;

    int *buff = (int *)malloc(buff_size*sizeof(int));
    assert(buff != NULL);

    int start = 0, end = 0;


    while(visited < n1){
        #pragma omp for schedule(static)
        for(int i = 0; i < n1; i++){
            int u = shellVert[i];
            if(flag[u] < 1){
                if(Dout[u] == level || upper[u] == level){
                    Dout[u] = level;
                    buff[end] = u;
                    end++;
                    flag[u] = 1;
                    core[u] = level;
                }
                else if (Din[u] < k){
                    Dout[u] = level;
                    buff[end] = u;
                    end++;
                    flag[u]++;
                    core[u] = level;
                }
            }
        }


        while(start < end){
            int v = buff[start];
            start++;

            for(int j = 0; j < g.outEdges[v].size(); j++){
                int u = g.outEdges[v][j];

                if(flag[u] == 0){

                    int din_u = __sync_fetch_and_sub(&Din[u], 1);
                    int expected = 0;
                    if(din_u == k && __atomic_compare_exchange_n(&flag[u], &expected, 1, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){
                        buff[end] = u;
                        end++;
                        core[u] = level;
                    }
                }

            }

            for(int j = 0; j < g.inEdges[v].size(); j++){
                int u = g.inEdges[v][j];


                if(flag[u] == 0 && Dout[u] > level){
                    int du = __sync_fetch_and_sub(&Dout[u], 1);
                    int expected = 0;
                    if(du==(level+1) && __atomic_compare_exchange_n(&flag[u], &expected, 1, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){
                        buff[end] = u;
                        end++;
                        core[u] = level;
                    }

                    if(du <= level){
                        __sync_fetch_and_add(&Dout[u], 1);
                    }
                }

            }
        }

        __sync_fetch_and_add(&visited, end);


        #pragma omp barrier

        #pragma omp for schedule(static)
        for(int i = 0; i < n; i++){
            if(flag[i] == 1 && core[i] == level){
                Dout[i] = level;
            }
        }

        #pragma omp barrier
        start = 0;
        end = 0;
        level = level+1;
        #pragma omp barrier

    }

    free( buff );
}


    return Dout;
}


vector<int> kshell(GraphCPU& g, vector<int> iH, vector<int>& kBin, int* vert){
	unsigned n = g.V;
	vector<int> pos;

	pos.resize(n+1);
	// vert.resize(n);
	kBin.resize(n);

	for (int i = 0; i < n; i++) {
		kBin[iH[i]] += 1;
	}

	int level = *max_element(iH.begin(),iH.end());

	int start = 0;
	for (int i = 0; i <= level; i++) {
		int num = kBin[i];
		kBin[i] = start;
		start += num;
	}
	for (int i = 0; i < n; i++) {
		pos[i] = kBin[iH[i]];
		vert[pos[i]] = i;
		kBin[iH[i]] += 1;
	}
	for (int i = level; i > 0; i--) {
		kBin[i] = kBin[i - 1];
	}
	kBin[0] = 0;


	int record = -1;
	int* result = new int [n];
	int resultPos = 0;
	for(int i = 0; i< n;i++){
		if(iH[vert[i]] != record){
			result[resultPos] = iH[vert[i]];
			resultPos ++;
			record  = iH[vert[i]];
		}
	}
	result[0] = -1;
	int maxK = record;
	for(int i = 1;i<resultPos;i++){
		result[i] = result[i-1]+1;
	}
	result[resultPos] = maxK;

	std::vector<int> res;
	for(int i=1;i<resultPos+1;i++){
		res.push_back(result[i]);
	}

	return res;
}


void PDC(GraphCPU& g){
	unsigned n = g.V;

    //获取线程数
    int NUM_THREADS = num_of_thread;
    omp_set_num_threads(NUM_THREADS);
    // #pragma omp parallel
    // {
    //     #pragma omp master
    //     NUM_THREADS = omp_get_num_threads();
    // }

    int vis_num = 0;

    std::vector<int> iH, OC;
    iH.resize(n);
    OC.resize(n);
    for (int i = 0; i < n; i++) {
        OC[i] = g.outDegrees[i];
        iH[i] = g.inDegrees[i];
    }

    // printf("number of threads: %d.\n", NUM_THREADS);


    // printf("in-deg size: %d, n: %d.\n",iH.size(),n);



    //每个线程运行这部分代码
#pragma omp parallel
{
    int level = 0;

    long buff_size = n;

    // printf("buff size: %d.\n", buff_size);

    int *buff = (int *)malloc(buff_size*sizeof(int));
    assert(buff != NULL);

    int start = 0, end = 0;

    while(vis_num < n){

        // printf("test.\n");

        //获取in-degree为level的顶点
        #pragma omp for schedule(static)
        for(int i = 0; i < n; i++){
            // printf("dddd.\n");
            if(iH[i] == level){
                buff[end] = i;
                end++;
            }
        }

        // printf("end: %d.\n", end);

        //处理buff中的顶点
        while(start < end){
            int v = buff[start];
            start++;

            for(int j = 0; j < g.outEdges[v].size(); j++){
                int u = g.outEdges[v][j];
                int din_u = iH[u];

                if(din_u > level){
                    int du = __sync_fetch_and_sub(&iH[u], 1);
                    //将下一个level的顶点放在buff末尾
                    if(du==(level+1)){
                        buff[end] = u;
                        end++;
                    }

                    if(du <= level) __sync_fetch_and_add(&iH[u], 1);
                }

            }
        }

        //累计处理过的顶点数
        __sync_fetch_and_add(&vis_num, end);

        #pragma omp barrier
        start = 0;
        end = 0;
        level = level+1;

        // printf("rest: %d.\n",n-vis_num);
        // printf("level: %d.\n", level);

    }


    free(buff);
}

    // printf("peeling done.\n");

    int level = *max_element(iH.begin(),iH.end());

    //找出k-shell对应的k
	vector<int> kBin;
	int* vert = (int *)malloc(n * sizeof(int));

    vector<int> res;
    res = kshell(g, iH, kBin, vert);

	// for (auto r: res)
	// 	cout << r << " ";
	// cout << endl;

    printf("kmax: %d, number of shells: %ld.\n", level, res.size());


    //对于shell中的每个k, 调用l的分解算法
    std::vector<std::vector<int>> Fres;

    // for (auto k : res) {
    //     Fres.push_back(Parpeel(k));
    // }
    Fres.push_back(Parpeel(g, 0, kBin, vert));
    // Fres.push_back(Parpeel(1));
    for(int i=1; i<res.size(); i++){
        Fres.push_back(Parpeel(g, res[i],Fres.back(), kBin, vert));
    }

    //write results: 1 row
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

    printf("decomposition done.\n");

	free(vert);
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

	PDC(g);

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
