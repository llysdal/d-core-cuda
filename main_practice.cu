#include <algorithm>
#include <iostream>
#include <ostream>
#include <vector>

// #include "graph.h"

using namespace std;

// int main(int argc, char *argv[]) {
// 	string filename = "../dataset/amazon0601.txt";
//
//     cout << "Graph loading Started... " << endl;
//     Graph g(filename);
//     cout << ">" << filename << endl;
//     cout << "V: " << g.V << endl;
//     cout << "E: " << g.E << endl;
// }


__global__ void vectorAdd(int *a, int *b, int *c, int n) {
	if (const unsigned id = (blockIdx.x * blockDim.x) + threadIdx.x; id < n)
		c[id] = a[id] + b[id];
}

__global__ void matrixMultiply(const int *a, const int *b, int *res, const unsigned n) {
	unsigned x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if ((x >= n) || (y >= n)) return;

	int sum = 0;
	for (int k = 0; k < n; k++) {
		sum += a[y * n + k] * b[k * n + x];
	}
	res[y * n + x] = sum;
}

#define SHMEM_SIZE (1 << 10)
__global__ void tiledMatrixMultiply(const int *a, const int *b, int *res, const unsigned n) {
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];

	unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned col = blockIdx.x * blockDim.x + threadIdx.x;

	int temp = 0;

	for (unsigned i = 0; i < n; i += blockDim.x) {
		A[threadIdx.y * blockIdx.x + threadIdx.x] = a[row * n + i + threadIdx.x];
		B[threadIdx.y * blockIdx.x + threadIdx.x] = b[i * n + threadIdx.y * n + col];

		__syncthreads();

		for (unsigned j = 0; j < blockDim.x; j++) {
			temp += A[threadIdx.y * blockDim.x + j] * B[j * blockDim.x + threadIdx.x];
		}

		__syncthreads();
	}
	res[row * n + col] = temp;
}

void init_vector(vector<int>& a, int n) {
	for (int i = 0; i < n; i++)
		a[i] = rand() % 100;
}
void init_array(int* a, int n) {
	for (int i = 0; i < n; i++)
		a[i] = rand() % 100;
}
void init_matrix(vector<int>& a) {
	ranges::generate(a, []() { return rand() % 100; });
}


int main() {
	srand(time(nullptr));

	int id;
	cudaGetDevice(&id);

	constexpr int n = 1 << 10;
	constexpr unsigned bytes = n*n*sizeof(int);

	// host matrix
	vector<int> h_a(n * n);
	vector<int> h_b(n * n);
	vector<int> h_c(n * n);

	// device memory
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	init_matrix(h_a);
	init_matrix(h_b);

	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);


	dim3 threads(32, 32);
	dim3 blocks(n/32, n/32);

	// cudaMemPrefetchAsync(a, N*sizeof(int), id, 0);
	// cudaMemPrefetchAsync(b, N*sizeof(int), id, 0);
	// vectorAdd<<<threads, blocks>>>(d_a, d_b, d_c, n);
	cout << "loaded" << endl;
	matrixMultiply<<<threads, blocks>>>(d_a, d_b, d_c, n);
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
	tiledMatrixMultiply<<<threads, blocks>>>(d_a, d_b, d_c, n);

	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
	// cudaDeviceSynchronize();
	cout << "done" << endl;

	//check correctness
	// int errors = 0;
	// // for (int i = 0; i < n; i++) {
	// // 	if (c[i] != a[i] + b[i])
	// // 		errors++;
	// // }
	// // For every row...
	// for (int i = 0; i < n; i++) {
	// 	// For every column...
	// 	for (int j = 0; j < n; j++) {
	// 		// For every element in the row-column pair
	// 		int tmp = 0;
	// 		for (int k = 0; k < n; k++) {
	// 			// Accumulate the partial results
	// 			tmp += h_a[i * n + k] * h_b[k * n + j];
	// 		}
	//
	// 		// Check against the CPU result
	// 		if (tmp != h_c[i * n + j])
	// 			errors++;
	// 	}
	// }
	// cout << "checked" << endl;
	//
	// cout << errors << endl;
}


// Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)   Category                   Operation
// --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -----------  ----------------------------------------
// 	59.8          2599783          2  1299891.5  1299891.5    437179   2162604    1220059.7  MEMORY_OPER  [CUDA memcpy Host-to-Device]
// 	30.8          1336756          1  1336756.0  1336756.0   1336756   1336756          0.0  CUDA_KERNEL  matrixMultiply(int *, int *, int *, int)
// 	 9.4           407581          1   407581.0   407581.0    407581    407581          0.0  MEMORY_OPER  [CUDA memcpy Device-to-Host]

// Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)   Category                        Operation
// --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -----------  --------------------------------------------------
// 	57.5          2587976          2  1293988.0  1293988.0    434684   2153292    1215239.4  MEMORY_OPER  [CUDA memcpy Host-to-Device]
// 	33.3          1497490          1  1497490.0  1497490.0   1497490   1497490          0.0  CUDA_KERNEL  tiledMatrixMultiply(int *, int *, int *, int, int)
// 	 9.2           414908          1   414908.0   414908.0    414908    414908          0.0  MEMORY_OPER  [CUDA memcpy Device-to-Host]