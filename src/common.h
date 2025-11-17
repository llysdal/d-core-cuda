#ifndef COMMON_H
#define COMMON_H

// #include <cuda_runtime_api.h>
// #include <cuda.h>
#include <cinttypes>
#include <omp.h>
#include <vector>

using namespace std;

// #define PRINT_STEPS
#define PRINT_MAINTENANCE_STATS

// #define HINDEX_NOWARP
#define HINDEX_WARP	//10x speedup!! (tested with email single add 47->46) [3x speedup on livejournal 2->1]
					//even more now that kmax also has warp!!

// #define PED_NOWARP
#define PED_WARP	//10x speedup for rly heavy batching :)

#define USE_RESTRICTIVE_KLIST_COMPUTE_MASK	// this only works for single insertions (until lmax batching is made :3)

#define SINGlE_INSERT_SKIP_CHECK	false	// check whether the single insert/delete is correct

#define FORCE_RECALCULATE_DCORE		false
#define FORCE_REBUILD_GRAPH			false	// required for non in-place insertion
#define OFFSET_GAP			1

#define BLOCK_COUNT			50
#define BLOCK_DIM			1024
#define WARPS_EACH_BLOCK	(BLOCK_DIM >> 5)
#define THREAD_COUNT		(BLOCK_DIM * BLOCK_COUNT)
#define WARP_COUNT			(WARPS_EACH_BLOCK * BLOCK_COUNT)
#define IS_MAIN_THREAD		threadIdx.x == 0
#define BLOCK_ID			blockIdx.x
#define THREAD_ID			threadIdx.x
#define GLOBAL_THREAD_ID	(BLOCK_ID * BLOCK_DIM + THREAD_ID)
#define WARP_SIZE			32
#define WARP_ID				(THREAD_ID >> 5)
#define GLOBAL_WARP_ID		(BLOCK_ID * WARPS_EACH_BLOCK + WARP_ID)
#define LANE_ID				(THREAD_ID & 31)
#define IS_MAIN_IN_WARP		(LANE_ID == 0)

#define BUFFER_SIZE			1'000'000

#define MODIFIED_EDGES_BUFFER_SIZE 20'000


typedef int degree;
typedef unsigned vertex;
typedef unsigned offset;

class GraphInterface {
public:
	unsigned V;
	unsigned E;
	degree kmax;
	vector<degree> kmaxes;
	degree lmax;
	vector<vector<degree>> lmaxes;
	virtual void insertEdges(const vector<pair<vertex, vertex>>& edgesToBeInserted) = 0;
	virtual void insertEdgesInPlace(const vector<pair<vertex, vertex>>& edgesToBeInserted) = 0;
	virtual pair<vertex, vertex> getRandomInsertEdge() = 0;
	virtual void deleteEdges(const vector<pair<vertex, vertex>>& edgesToBeDeleted) = 0;
	virtual void deleteEdgesInPlace(const vector<pair<vertex, vertex>>& edgesToBeDeleted) = 0;
	virtual ~GraphInterface() {}
};

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
	swap(d_p.in_degrees, d_p.out_degrees);
	swap(d_p.in_degrees_orig, d_p.out_degrees_orig);
	swap(d_p.in_neighbors, d_p.out_neighbors);
	swap(d_p.in_neighbors_offset, d_p.out_neighbors_offset);
}

#endif //COMMON_H