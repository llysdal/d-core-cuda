#ifndef D_CORE_CUDA_COMMON_CUH
#define D_CORE_CUDA_COMMON_CUH

#include "common.h"

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

#endif //D_CORE_CUDA_COMMON_CUH