#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <ctime>
#include <omp.h>
#include <limits>
#include <numeric>
// #include <sys/time.h>
#include <string.h>
#include "Edge.h"



class Graph_v2_test {
public:
    explicit Graph_v2_test(FILE* file, int t);

    explicit Graph_v2_test(int node_num, const std::vector<std::pair<int, int> > &edges);

    Graph_v2_test(const Graph_v2_test& g);

    ~Graph_v2_test();

	void PDC();
	void PDC_org();

    string ds;
private:

    int n;
    double m;
    int lmax = 0;
    int kmax = 0;
    int dmax = 0;
    long long iterations = 0;


    int num_of_thread;
    std::vector<std::vector<int>> *adj {nullptr};

    std::vector<std::vector<int>> uadj;

    // vector<vector<int>> in_adj;
    // vector<vector<int>> in_adj;

    std::vector<int> *deg {nullptr};
    int *outCdeg {nullptr};
    int *inCdeg {nullptr};

    int *outTdeg {nullptr};
    int *inTdeg {nullptr};


    
    std::vector<int> max_deg;

    std::vector<int> iH;

    std::vector<int> oH;

    std::vector<int> uH;

    std::vector<int> OC;

    std::vector<int> *iHk {nullptr};

    std::vector<int> *iHk1 {nullptr};

    std::vector<int> *sH {nullptr}; //slyline core number
    std::vector<int> *tmp_results {nullptr}; //slyline core number
    
    omp_lock_t *lock; 

    // std::vector<bool> changed; //for each vertex
    bool *changed {nullptr}; //for each vertex

    double cost_mt {0}; // mt: cost of maintain the in-degree of each (k, l)-core with a given k.
    double cost_fd {0}; // fd: cost of find the minimal in-degree of each (k, l)-core with a given k.
    double cost_d {0}; // d: cost of decomposition.
    // struct timeval begin_mt, end_mt;
    // struct timeval begin_fd, end_fd;
    // struct timeval begin_d, end_d;

    void add_edge(int u, int v);

    void sort_adj_list();

    void init_adj_deg();


    /////////// for peeling ////////////
    int kCoreMax = 0;
    int Kmd;

    //Kdeg stores all vertices's max possible K, where there is a (K,0)-core contains it
    //kBin is used to do a bucket sort for all vertices in ascending order of their Kdeg
    std::vector<int> Kdeg, kBin; //used to calculate all (k,0)-cores

    std::vector<int> bin, pos; 
    int *vert {nullptr};//used to calculate rows
    //give a certain K,
    //bin is a bucked sort in ascending number of vertices' max L where there is a (K,L)-core contains it
    //pos: which bin a vertex is in
    //vert: the array of all vertices in this row(namely in (K,0)-core)

    std::vector<int> subLDegree, subKDegree;           //record a vertex 's out and in degree in current subgraph_v1
    std::vector<bool> visited, isInSub;     //isInsub used in getRow, judge whether a vetex satisfy the basic k constraint (namely in (k,0)-core)

    //we do D-core decomposition row by row
    //one row is, a certain K, all the (K, l) cores sorted by ascending order of l
    std::vector<int> rowVert;                          //store vertices in one row                   
    int rowResultPos;                    //a mark variable used to fill a row result into rowResult[]
    std::vector<int> sorted_ID;                //a strictly sorted array, sorted by out-core number
    std::vector<int> rowResult;                //a strictly sorted array, sorted by 1 core number
    std::vector<int> compulsoryVisit;
    int compulsorySubL,compulsoryNum;

	// parallel peeling
	std::vector<int> Parpeel(int k);
	std::vector<int> Parpeel(int k, std::vector<int> upper);
	std::vector<int> kshell(std::vector<int> iH);

	std::vector<int> Parpeel_org(int k);
};
