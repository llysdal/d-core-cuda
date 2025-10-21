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
			outfile << setw(width);
			outfile << v << " ";
		}
		outfile << "\r\n";
	}
	auto endWrite = chrono::steady_clock::now();
	cout << "D-core results written\t\t" << chrono::duration_cast<chrono::milliseconds>(endWrite - startWrite).count() << "ms" << endl;
}


#endif //D_CORE_CUDA_UTILS_H