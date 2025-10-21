#ifndef D_CORE_CUDA_CUDAMEM_CUH
#define D_CORE_CUDA_CUDAMEM_CUH

// Graph memory
void allocateDeviceGraphMemory(Graph& g, device_graph_pointers& g_p) {
	cudaMalloc(&(g_p.in_neighbors), g.E * sizeof(vertex));
	cudaMalloc(&(g_p.out_neighbors), g.E * sizeof(vertex));
	cudaMalloc(&(g_p.in_neighbors_offset), (g.V+1) * sizeof(offset));
	cudaMalloc(&(g_p.out_neighbors_offset), (g.V+1) * sizeof(offset));
	cudaMalloc(&(g_p.in_degrees), (g.V) * sizeof(degree));
	cudaMalloc(&(g_p.out_degrees), (g.V) * sizeof(degree));
}

void deallocateDeviceGraphMemory(device_graph_pointers g_p) {
	cudaFree(g_p.in_neighbors);
	cudaFree(g_p.out_neighbors);
	cudaFree(g_p.in_neighbors_offset);
	cudaFree(g_p.out_neighbors_offset);
	cudaFree(g_p.in_degrees);
	cudaFree(g_p.out_degrees);
}

// Accessory memory (used for peeling)
void allocateDeviceAccessoryMemory(Graph& g, device_accessory_pointers& a_p) {
	cudaMalloc(&(a_p.buffers), BUFFER_SIZE * BLOCK_NUMS * sizeof(vertex));
	cudaMalloc(&(a_p.bufferTails), BLOCK_NUMS * sizeof(unsigned));
	cudaMalloc(&(a_p.global_count), sizeof(unsigned));
	cudaMalloc(&(a_p.visited), g.V * sizeof(unsigned));	// unsigned instead of bool since atomicCAS only doesn't do bools
	cudaMalloc(&(a_p.core), g.V * sizeof(degree));
}

void deallocateDeviceAccessoryMemory(device_accessory_pointers a_p) {
	cudaFree(a_p.buffers);
	cudaFree(a_p.bufferTails);
	cudaFree(a_p.global_count);
	cudaFree(a_p.visited);
	cudaFree(a_p.core);
}

// Maintenance memory (used for d-core maintenance)
void allocateDeviceMaintenanceMemory(Graph& g, device_maintenance_pointers& m_p) {
	cudaMalloc(&(m_p.k_max), g.V * sizeof(degree));
	cudaMalloc(&(m_p.original_k_max), g.V * sizeof(degree));
	cudaMalloc(&(m_p.l_max), g.V * sizeof(degree));
	cudaMalloc(&(m_p.original_l_max), g.V * sizeof(degree));
	cudaMalloc(&(m_p.new_l_max), g.V * sizeof(degree));
	cudaMalloc(&(m_p.compute), g.V * sizeof(unsigned));
	cudaMalloc(&(m_p.ED), g.V * sizeof(degree));
	cudaMalloc(&(m_p.PED), g.V * sizeof(degree));
	cudaMalloc(&(m_p.flag), sizeof(bool));
	cudaMalloc(&(m_p.tmp_neighbor_coreness), g.E * sizeof(degree));
	cudaMalloc(&(m_p.hIndex_buckets), (g.V + g.E) * sizeof(degree));
	cudaMalloc(&(m_p.histograms), (g.V + g.E) * sizeof(unsigned));
}

void deallocateDeviceMaintenanceMemory(device_maintenance_pointers& m_p) {
	cudaFree(m_p.k_max);
	cudaFree(m_p.original_k_max);
	cudaFree(m_p.l_max);
	cudaFree(m_p.original_l_max);
	cudaFree(m_p.new_l_max);
	cudaFree(m_p.compute);
	cudaFree(m_p.ED);
	cudaFree(m_p.PED);
	cudaFree(m_p.flag);
	cudaFree(m_p.tmp_neighbor_coreness);
	cudaFree(m_p.hIndex_buckets);
	cudaFree(m_p.histograms);
}

#endif //D_CORE_CUDA_CUDAMEM_CUH