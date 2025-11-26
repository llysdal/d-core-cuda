#ifndef D_CORE_CUDA_UTILS_H
#define D_CORE_CUDA_UTILS_H
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"

using namespace std;

bool compareDCoreResults(const string& first, const string& second) {
	auto startCompare = chrono::steady_clock::now();
	string result = "equal";
	ifstream f1(first, ios::binary|ios::ate);
	ifstream f2(second, ios::binary|ios::ate);

	if (f1.fail() || f2.fail()) {
		result = "file problem";
		auto endCompare = chrono::steady_clock::now();
		cout << "D-core results compared\t\t" << chrono::duration_cast<chrono::milliseconds>(endCompare - startCompare).count() << "ms" << endl;
		cout << "\t" << result << endl;
		return false;
	}
	if (f1.tellg() != f2.tellg()) {
		result = "size mismatch";
		auto endCompare = chrono::steady_clock::now();
		cout << "D-core results compared\t\t" << chrono::duration_cast<chrono::milliseconds>(endCompare - startCompare).count() << "ms" << endl;
		cout << "\t" << result << endl;
		return false;
	}

	f1.seekg(0, ios::beg);
	f2.seekg(0, ios::beg);
	if (!equal(istreambuf_iterator<char>(f1.rdbuf()), istreambuf_iterator<char>(), istreambuf_iterator<char>(f2.rdbuf())))
		result = "not equal";

	auto endCompare = chrono::steady_clock::now();
	cout << "D-core results compared\t\t" << chrono::duration_cast<chrono::milliseconds>(endCompare - startCompare).count() << "ms" << endl;
	cout << "\t" << result << endl;
	return result == "equal";
}

void writeDCoreResults(vector<vector<degree>>& values, const string& outputFile) {
	auto startWrite = chrono::steady_clock::now();
	ofstream bin;
	bin.open(outputFile, ios::binary|ios::out|ios::trunc);
	if (bin) {
		for (const auto r: values)
			bin.write(reinterpret_cast<const char*>(r.data()), static_cast<streamsize>(r.size() * sizeof(degree)));
	} else {
		cout << outputFile << ": could not open file" << endl;
	}
	auto endWrite = chrono::steady_clock::now();
	cout << "D-core results written\t\t" << chrono::duration_cast<chrono::milliseconds>(endWrite - startWrite).count() << "ms" << endl;
}

void writeDCoreResultsText(vector<vector<degree>>& values, const string& outputFile, unsigned lmax) {
	auto startWrite = chrono::steady_clock::now();
	// this is for writing to text files
	long long width = to_string(lmax).length();
	ofstream outfile (outputFile,ios::out|ios::binary|ios::trunc);
	for (auto r: values) {
		for (auto v: r) {
			// outfile << setw(width); not supported
			outfile << v << " ";
		}
		outfile << "\r\n";
	}
	auto endWrite = chrono::steady_clock::now();
	cout << "D-core results written\t\t" << chrono::duration_cast<chrono::milliseconds>(endWrite - startWrite).count() << "ms" << endl;
}


bool readDecompFile(Graph& g, const string& filename) {
	ifstream decompin;
	decompin.open(filename + ".decomp", ios::binary | ios::in);
	if (!decompin || FORCE_RECALCULATE_DCORE) {
		return false;
	}
	cout << "Using decomp file..." << endl;
	auto decompReadStart = chrono::steady_clock::now();
	decompin.read(reinterpret_cast<char*>(&g.kmax), sizeof(unsigned));
	decompin.read(reinterpret_cast<char*>(&g.lmax), sizeof(unsigned));
	decompin.read(reinterpret_cast<char*>(g.kmaxes.data()), static_cast<streamsize>(g.V * sizeof(degree)));
	g.lmaxes.resize(g.kmax+1);
	for (unsigned k = 0; k <= g.kmax; ++k) {
		g.lmaxes[k] = vector<degree>(g.V);
		decompin.read(reinterpret_cast<char*>(g.lmaxes[k].data()), static_cast<streamsize>(g.V * sizeof(degree)));
	}
	decompin.close();
	cout << "Read decomp file\t\t" << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - decompReadStart).count() << "ms" << endl;
	cout << "\tkmax: " << g.kmax << endl;
	cout << "\tlmax: " << g.kmax << endl;
	return true;
}
void writeDecompFile(Graph& g, const string& filename) {
	ofstream decompout;
	decompout.open(filename + ".decomp", ios::binary | ios::out);
	decompout.write(reinterpret_cast<char*>(&g.kmax), sizeof(unsigned));
	decompout.write(reinterpret_cast<char*>(&g.lmax), sizeof(unsigned));
	decompout.write(reinterpret_cast<char*>(g.kmaxes.data()), static_cast<streamsize>(g.V * sizeof(degree)));
	for (unsigned k = 0; k <= g.kmax; ++k)
		decompout.write(reinterpret_cast<char*>(g.lmaxes[k].data()), static_cast<streamsize>(g.V * sizeof(degree)));
	decompout.close();
}
void writeDecompFileCPU(Graph& g, const string& filename) {
	ofstream decompout;
	decompout.open(filename + ".decompCPU", ios::out);
	decompout << g.V << endl;;
	for (unsigned v = 0; v < g.V; ++v) {
		unsigned end = g.kmaxes[v];
		for (unsigned j = 0; j <= end; ++j) {
			decompout << g.lmaxes[j][v];
			if (j < end) decompout << " ";
		}
		if (v != g.V-1)
			decompout << endl;
	}
	decompout.close();
}


#define DIM(x) (sizeof(x)/sizeof(*(x)))

static const char     *sizes[]   = { "EiB", "PiB", "TiB", "GiB", "MiB", "KiB", "B" };
static const uint64_t  exbibytes = 1024ULL * 1024ULL * 1024ULL *
								   1024ULL * 1024ULL * 1024ULL;
char* calculateSize(uint64_t size) {
	char* result = (char *) malloc(sizeof(char) * 20);
	uint64_t multiplier = exbibytes;
	int i;

	for (i = 0; i < DIM(sizes); i++, multiplier /= 1024) {
		if (size < multiplier)
			continue;
		if (size % multiplier == 0)
			sprintf(result, "%" PRIu64 " %s", size / multiplier, sizes[i]);
		else
			sprintf(result, "%.1f %s", (float) size / multiplier, sizes[i]);
		return result;
	}
	strcpy(result, "0");
	return result;
}

#endif //D_CORE_CUDA_UTILS_H