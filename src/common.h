#ifndef COMMON_H
#define COMMON_H

// #include <cuda_runtime_api.h>
// #include <cuda.h>
#include <cinttypes>
#include <omp.h>
#include <vector>

// #define PRINT_STEPS
#define PRINT_MAINTENANCE_STATS

#define HINDEX_NOWARP
// #define HINDEX_WARP	//10x speedup!! (tested with email single add 47->46) [3x speedup on livejournal 2->1]

#define SINGlE_INSERT_SKIP_CHECK	true	// check whether the single insert is correct

#define FORCE_RECALCULATE_DCORE		false
#define FORCE_REBUILD_GRAPH			false	// required for non in-place insertion
#define OFFSET_GAP			1

#define BLOCK_NUMS			50
#define BLOCK_DIM			1024
#define WARPS_EACH_BLOCK	(BLOCK_DIM >> 5)
#define THREAD_COUNT		(BLOCK_DIM * BLOCK_NUMS)
#define WARP_COUNT			(WARPS_EACH_BLOCK * BLOCK_NUMS)
#define IS_MAIN_THREAD		threadIdx.x == 0
#define THREAD_ID			threadIdx.x
#define WARP_SIZE			32
#define WARP_ID				(THREAD_ID >> 5)
#define LANE_ID				(THREAD_ID & 31)
#define IS_MAIN_IN_WARP		(LANE_ID == 0)

#define BUFFER_SIZE		1'000'000

#define MODIFIED_EDGES_BUFFER_SIZE 2'000


typedef int degree;
typedef unsigned vertex;
typedef unsigned offset;

typedef struct GraphData {
	unsigned V;
	std::vector<degree>& kmaxes;
	std::vector<std::vector<degree>>& lmaxes;
} GraphData;

typedef struct device_graph_pointers {
	vertex* in_neighbors;
	vertex* out_neighbors;
	offset* in_neighbors_offset;
	offset* out_neighbors_offset;
	degree* in_degrees;
	degree* out_degrees;
	degree* in_degrees_orig;	// we need these origs since the others are clobbered and we might have offsets built in
	degree* out_degrees_orig;
	vertex*	modified_edges;
} device_graph_pointers;

typedef struct device_accessory_pointers {
	vertex*		buffers;		// each block has a buffer of size BUFFER_SIZE
	unsigned*	bufferTails;	// each block has a buffer tail for keeping track of where to write
	unsigned*	global_count;	// this is the total amount of processed vertices across all blocks
	unsigned*	visited;		// the set of processed nodes - we only process each node once
	degree*		core;			// the resulting l-values (?) to form the k-list
} device_accessory_pointers;

typedef struct device_maintenance_pointers {
	degree*		k_max_max;
	degree*		m_value;
	degree*		k_max;
	degree*		l_max;
	degree*		new_l_max;
	unsigned*	compute;
	degree*		ED;
	degree*		PED;
	bool*		flag;
	unsigned*	histograms;
} device_maintenance_pointers;

inline void swapInOut(device_graph_pointers& d_p) {
	// this is an easy way to turn our KList function into an LList function!
	std::swap(d_p.in_degrees, d_p.out_degrees);
	std::swap(d_p.in_degrees_orig, d_p.out_degrees_orig);
	std::swap(d_p.in_neighbors, d_p.out_neighbors);
	std::swap(d_p.in_neighbors_offset, d_p.out_neighbors_offset);
}

#endif //COMMON_H