#ifndef D_CORE_CUDA_COMMON_CUH
#define D_CORE_CUDA_COMMON_CUH

#include "common.h"

__device__ inline unsigned readFromBuffer(unsigned* buffer, unsigned loc) {
	assert(loc < BUFFER_SIZE);
	return buffer[loc];
}
__device__ inline void writeToBuffer(unsigned* buffer, unsigned loc, unsigned val) {
	assert(loc < BUFFER_SIZE);
	buffer[loc] = val;
}

#endif //D_CORE_CUDA_COMMON_CUH