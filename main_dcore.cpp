#include <iostream>
#include "src-dcore/Graph_v2_test.h"
// #include "Graph_v2.h"
// #include "Graph_v3.h"
// #include "Graph_v4.h"
#include <string>
#include <chrono>
// #include "cmdline.h"
// #include <sys/time.h>
#include <algorithm>
#include <omp.h>
using namespace std;


// l: out, k: in;
// 0: out, 1: in; 
int main(int argc, char *argv[]) {
    // //==============================================
    // // input parameters
    // cmdline::parser a;
    // a.add<string>("file", 'f', "filename", true, "");
    // a.add<int>("a", 'a', "algorithm", true);
    // a.add<string>("data", 'd', "dataset", true, "");
    // a.add<int>("t", 't', "threads", true);
  
    // a.parse_check(argc, argv);

    // // Read graph
    // string filepath = a.get<string>("file");
    // string ds = a.get<string>("data");
    // int type = a.get<int>("a");
    // int t = a.get<int>("t");
    // FILE* dFile = fopen(filepath.c_str(), "r");
    // //==============================================



    // // =============================================
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/em.txt";
    // string ds = "em";

    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/sd.txt";
    // string ds = "sd";
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/am.txt";
    // string ds = "am";
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/em.txt";
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/po.txt";
    // string ds = "po";
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/cite.txt";
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/lj.txt";
    // string ds = "lj";
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/hw.txt";
    // string ds = "hw";
    string filepath = "../dataset/congress.txt";
	string ds = "cg";
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/it.txt";
    // string ds = "it";
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/ew2013.txt";
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/uk2007.txt";
    // string filepath = "/home/chunxu/cxx/Dcore_tool/materials/scalability/em/em20.txt";
    FILE* dFile = fopen(filepath.c_str(), "r");

    int type = 8;
    // int type = 13;
    // int type = 7;
    // int type = 9;
    // int type = 1;
    int t = 1;
    // // =============================================
    // printf("file: %s; \t algorithm: %d; \t threads: %d\n", filepath.c_str(), type, t);

    // clock_t io_begin = clock();

	cout << "loading graph" << endl;
    Graph_v2_test g = Graph_v2_test(dFile,t);
	g.ds = ds;
	cout << "loaded graph" << endl;
    // Graph_v2 g = Graph_v2(dFile,t);
    // Graph_v3 g = Graph_v3(dFile,t);
    // Graph_v4 g = Graph_v4(dFile,t);

    // clock_t io_end = clock();
    // double io_secs = double(io_end - io_begin) / CLOCKS_PER_SEC;
    // printf("io time: %.4f.\n", io_secs);


    // struct timeval begin, end;
    // gettimeofday(&begin, NULL);
    // auto begin = chrono::steady_clock::now();
    //clock_t begin = clock();

	cout << "running algo" << endl;
    g.PDC_org();
	cout << "done" << endl;

    //clock_t end = clock();
    // auto end = chrono::steady_clock::now();
    // gettimeofday(&end, NULL);
    // double runtime = (end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec)/1000000.0;
    // printf("running time: %.4f sec.\n", runtime);

    
    //g.output_ds();

    return 0;
}