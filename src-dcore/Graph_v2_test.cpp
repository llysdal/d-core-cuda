#include "../src-dcore/Graph_v2_test.h"
#include "omp.h"
#include <cassert>
#include <fstream>
#include <chrono>
#include <iomanip>

// export OMP_WAIT_POLICY=passive

Graph_v2_test::Graph_v2_test(FILE* file, int t) {
    num_of_thread = t;
    // clock_t begin = clock();
    fscanf(file, "# %d", &n);
	// cout << n << endl;
    init_adj_deg();

    m = 0;
	int u, v;
    while (fscanf(file, "%d%d", &u, &v) == 2)
        add_edge(u, v);

    sort_adj_list();

    // clock_t end = clock();
    // double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    // printf("graph_v1 construction: %.4f\n", elapsed_secs);
}

void Graph_v2_test::init_adj_deg() {
    max_deg.resize(2, 0);
    iHk = new std::vector<int>[n]; //for each vertex
    iHk1 = new std::vector<int>[n]; //for each vertex
    sH = new std::vector<int>[n];
    tmp_results = new std::vector<int>[n];
    lock = new omp_lock_t[n];
    for(int i = 0; i < n; i++){
        omp_init_lock(&lock[i]);
    }

    adj = new std::vector<std::vector<int>>[2]; // 0 out edges, 1 in edges
    deg = new std::vector<int>[2]; // 0 out-degree, 1 in-degree
    m = 0;
    for (int i = 0; i < 2; i++) {
        adj[i].resize(static_cast<unsigned long>(n));
        deg[i].resize(static_cast<unsigned long>(n), 0);
    }
    // changed.resize(n,false);
    changed = (bool *)malloc(n * sizeof(bool));
}

void Graph_v2_test::sort_adj_list() {
	for (int i = 0; i < 2; i++) {
		for (int v = 0; v < n; v++) {
			std::sort(adj[i][v].begin(), adj[i][v].end(),
					  [&](const int& a, const int& b) -> bool
					  {
						  return deg[1 - i][a] > deg[1 - i][b];
					  });
		}
	}
}

inline void Graph_v2_test::add_edge(int u, int v) {
	adj[0][u].push_back(v);
	deg[0][u] += 1;
	adj[1][v].push_back(u);
	deg[1][v] += 1;
	max_deg[0] = std::max(max_deg[0], deg[0][u]);
	max_deg[1] = std::max(max_deg[1], deg[1][v]);
	++m;
}

Graph_v2_test::~Graph_v2_test() {
    for (int i = 0; i < 2; i++) {
        adj[i].clear();
        deg[i].clear();
    }
    delete[] adj;
    delete[] deg;
    max_deg.clear();
    for(int i = 0; i < n; i++){
        omp_destroy_lock(&lock[i]);
    }
    delete[] sH;
    delete[] tmp_results;
    delete[] lock;
    free(changed);
}




//parallel peeling for a given k, advanced version
std::vector<int> Graph_v2_test::Parpeel(int k){
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
        OC[i] = deg[0][i];
        iH[i] = deg[1][i];
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
        for (int l = 0; l < adj[0][v].size(); l++) {
            if (flag[adj[0][v][l]]) OC[v]--;
        }
        for (int k = 0; k < adj[1][v].size(); k++) {
            if (flag[adj[1][v][k]]) iH[v]--;
        }
    }

    int n1 = shellVert.size();

    // printf("n1: %d.\n",n1);


    // //获取每个顶点的OC的上界
    // if(k == 0)  upper = OC;



    //每个线程运行这部分代码
#pragma omp parallel
{
    int level = 0;

    long buff_size = n;

    int *buff = (int *)malloc(buff_size*sizeof(int));
    assert(buff != NULL);

    int start = 0, end = 0;

    // #pragma omp for schedule(static)
    int test = 0;
    // for(int i = 0; i < 10000000000;i++){
    //     for(int j=0;j<10000000;j++)
    //         test++;
    //     // if(flag[i] < 1){
    //     //     if (iH[i] < k){
    //     //         OC[i] = level;
    //     //         flag[i]++;
    //     //     }
    //     // }
    // }

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

            // for(int i = 0; i < 10000000000;i++)
            //     for(int j=0;j<10000000;j++)
            //         test++;

        }

        //获取out-degree为level的顶点, 或in-degree 小于k的顶点
        // #pragma omp for schedule(static)
        // for(int i = 0; i < n; i++){
        //     if(flag[i] < 1){
        //         if(OC[i] == level){
        //             buff[end] = i;
        //             end++;
        //             flag[i]++;
        //         }
        //         else if (iH[i] < k){
        //             OC[i] = level;
        //             buff[end] = i;
        //             end++;
        //             flag[i]++;
        //         }
        //     }
        // }

        // printf("level: %d, end: %d.\n", level, end);


        //处理buff中的顶点, 需要给小于din小于k的顶点加一个flag, 首先初始化所有顶点的flag为0,如果小于k, 那就将其修改为1, 并将其放入buff中
        while(start < end){
            int v = buff[start];
            start++;
            flag[v]++;

            //处理in-degree小于k的顶点
            for(int j = 0; j < adj[0][v].size(); j++){
                int u = adj[0][v][j];

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
            for(int j = 0; j < adj[1][v].size(); j++){
                int u = adj[1][v][j];

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
std::vector<int> Graph_v2_test::Parpeel(int k, std::vector<int> upper){
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
        Dout[i] = deg[0][i];
        Din[i] = deg[1][i];
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
        for (int l = 0; l < adj[0][v].size(); l++) {
            if (flag[adj[0][v][l]]) Dout[v]--;
        }
        for (int k = 0; k < adj[1][v].size(); k++) {
            if (flag[adj[1][v][k]]) Din[v]--;
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

            for(int j = 0; j < adj[0][v].size(); j++){
                int u = adj[0][v][j];

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

            for(int j = 0; j < adj[1][v].size(); j++){
                int u = adj[1][v][j];


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

//obtain all distinct k
std::vector<int> Graph_v2_test::kshell(std::vector<int> iH){
	pos.resize(n+1);
	// vert.resize(n);
	vert = (int *)malloc(n * sizeof(int));
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

void Graph_v2_test::PDC(){
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
        OC[i] = deg[0][i];
        iH[i] = deg[1][i];
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

            for(int j = 0; j < adj[0][v].size(); j++){
                int u = adj[0][v][j];
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
    std::vector<int> res;
    res = kshell(iH);

	// for (auto r: res)
	// 	cout << r << " ";
	// cout << endl;

    printf("kmax: %d, number of shells: %ld.\n", level, res.size());

    //对于shell中的每个k, 调用l的分解算法
    std::vector<std::vector<int>> Fres;

    // for (auto k : res) {
    //     Fres.push_back(Parpeel(k));
    // }
    Fres.push_back(Parpeel(0));
    // Fres.push_back(Parpeel(1));
    for(int i=1; i<res.size(); i++){
        Fres.push_back(Parpeel(res[i],Fres.back()));
    }

    int count4 = 0;

    //write results: 1 row
    std::ofstream outfile ("./res.txt",ios::in|ios::out|ios::binary|ios::trunc);
	// for(int i = 0; i < Fres.size(); i++) {
	// 	for(int j = 0; j < Fres[i].size(); j++){
	// 		outfile << Fres[i][j] << " ";
 //            if(Fres[i][j]==1)
 //                count4++;
	// 	}
	// 	outfile << "\r\n";
	// }
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

    // printf("count 4: %d.\n",count4);

    printf("decomposition done.\n");

    // std::vector<int> res_in = Parpeel_org(1);

    // int count4 = 0;

    // //write
    // std::ofstream outfile ("/mnt/data/wensheng/directed/dres/res1.txt",ios::in|ios::out|ios::binary|ios::trunc);
	// for(int i = 0; i < n; i++) {
	// 	outfile << res_in[i] << " ";
    //     if(res_in[i]==1)
    //         count4++;
	// }
    // outfile << "\r\n";

    // printf("count 4: %d.\n",count4);
    free(vert);

}