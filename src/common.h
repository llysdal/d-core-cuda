#ifndef CUDA_PLAYGROUND_COMMON_H
#define CUDA_PLAYGROUND_COMMON_H

// #include <cuda_runtime_api.h>
// #include <cuda.h>
#include <omp.h>

#define BLOCK_NUMS		32
#define BLOCK_DIM		1024
#define WARPS_EACH_BLOCK	(BLOCK_DIM >> 5)
#define THREAD_COUNT	(BLOCK_DIM * BLOCK_NUMS)
#define IS_MAIN_THREAD	threadIdx.x == 0
#define THREAD_ID		threadIdx.x
#define WARP_SIZE		32
#define WARP_ID			(THREAD_ID >> 5)
#define LANE_ID			(THREAD_ID & 31)

#define BUFFER_SIZE		1'000'000

typedef struct device_pointers {
	unsigned* in_neighbors;
	unsigned* out_neighbors;
	unsigned* in_neighbors_offset;
	unsigned* out_neighbors_offset;
	unsigned* in_degrees;
	unsigned* out_degrees;
} device_pointers;


#endif //CUDA_PLAYGROUND_COMMON_H