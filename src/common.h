#ifndef COMMON_H
#define COMMON_H

// #include <cuda_runtime_api.h>
// #include <cuda.h>
#include <omp.h>

#define BLOCK_NUMS		40
#define BLOCK_DIM		1024
#define WARPS_EACH_BLOCK	(BLOCK_DIM >> 5)
#define THREAD_COUNT	(BLOCK_DIM * BLOCK_NUMS)
#define IS_MAIN_THREAD	threadIdx.x == 0
#define THREAD_ID		threadIdx.x
#define WARP_SIZE		32
#define WARP_ID			(THREAD_ID >> 5)
#define LANE_ID			(THREAD_ID & 31)

#define BUFFER_SIZE		1'000'000

typedef int degree;
typedef unsigned vertex;
typedef unsigned offset;

typedef struct device_pointers {
	vertex* in_neighbors;
	vertex* out_neighbors;
	offset* in_neighbors_offset;
	offset* out_neighbors_offset;
	degree* in_degrees;
	degree* out_degrees;
} device_pointers;

inline void swapInOut(device_pointers& d_p) {
	std::swap(d_p.in_degrees, d_p.out_degrees);
	std::swap(d_p.in_neighbors, d_p.out_neighbors);
	std::swap(d_p.in_neighbors_offset, d_p.out_neighbors_offset);
}

#endif //COMMON_H