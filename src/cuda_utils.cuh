#ifndef D_CORE_CUDA_CUDA_UTILS_H
#define D_CORE_CUDA_CUDA_UTILS_H
#include "common.h"

// for decomp
__device__ inline vertex readVertexFromBuffer(vertex* buffer, unsigned loc) {
	assert(loc < BUFFER_SIZE);
	return buffer[loc];
}
__device__ inline void writeVertexToBuffer(vertex* buffer, unsigned loc, vertex val) {
	assert(loc < BUFFER_SIZE);
	buffer[loc] = val;
}

__device__ inline bool atomicTestAndSet(unsigned* adr) {
	return !atomicCAS(adr, 0, 1);
}

// __device__ inline short atomicAdd(short* address, short val) {
// 	short assumed = *address;
// 	short old = assumed;
// 	do
// 	{
// 		assumed = old;
// 		old = atomicCAS((unsigned short*)address, assumed, assumed + val);
// 	} while (assumed != old);
//
// 	return old;
// }
//
// __device__ inline short atomicSub(short* address, short val) {
// 	return atomicAdd(address, -val);
// }
//
// __device__ inline short int atomicMax(short int* address, short int val) {
// 	short assumed = *address;
// 	short old = assumed;
// 	do
// 	{
// 		assumed = old;
// 		old = atomicCAS((unsigned short*)address, assumed, assumed > val ? assumed : val);
// 	} while (assumed != old);
//
// 	return old;
// }

// for maintenance
__device__ degree hOutIndex(device_maintenance_pointers m_p, vertex v, offset o, degree upperBound) {
	offset histogramStart = o + v;

	degree cnt = 0;
	for (int i = upperBound; i >= 0; i--) {
		cnt += m_p.histograms[histogramStart + i];
		if (cnt >= i) return i;
	}
	return upperBound;
}

__device__ degree hInIndex(device_maintenance_pointers m_p, vertex v, offset o, degree upperBound, degree k) {
	offset histogramStart = o + v;

	degree cnt = 0;
	for (int i = upperBound; i >= 0; i--) {
		cnt += m_p.histograms[histogramStart + i];
		if (cnt >= k) return i;
	}
	return upperBound;
}

// for checking (not verified)
__global__ void compareArray(unsigned* errors, degree* a, degree* b, unsigned V) {
	if (IS_MAIN_THREAD)
		*errors = 0;

	__syncthreads();

	for (unsigned base = 0; base < V; base += WARP_COUNT) {
		vertex v = base + GLOBAL_WARP_ID;
		if (v >= V) break;

		if (a[v] != b[v])
			atomicAdd(errors, 1);
	}
}


#endif //D_CORE_CUDA_CUDA_UTILS_H