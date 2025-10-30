#ifndef D_CORE_CUDA_CUDAMEM_CUH
#define D_CORE_CUDA_CUDAMEM_CUH

// Graph memory
void allocateDeviceGraphMemory(Graph& g, device_graph_pointers& g_p) {
	// cudaMalloc(&(g_p.in_neighbors), g.E * sizeof(vertex));
	// cudaMalloc(&(g_p.out_neighbors), g.E * sizeof(vertex));
	cudaMalloc(&(g_p.in_neighbors), (g.V * OFFSET_GAP) * sizeof(vertex));
	cudaMalloc(&(g_p.out_neighbors), (g.V * OFFSET_GAP) * sizeof(vertex));
	cudaMalloc(&(g_p.in_neighbors_offset), (g.V+1) * sizeof(offset));
	cudaMalloc(&(g_p.out_neighbors_offset), (g.V+1) * sizeof(offset));
	cudaMalloc(&(g_p.in_degrees), (g.V) * sizeof(degree));
	cudaMalloc(&(g_p.out_degrees), (g.V) * sizeof(degree));
	cudaMalloc(&(g_p.modified_edges), MODIFIED_EDGES_BUFFER_SIZE * sizeof(vertex));
}

void deallocateDeviceGraphMemory(device_graph_pointers& g_p) {
	cudaFree(g_p.in_neighbors);
	cudaFree(g_p.out_neighbors);
	cudaFree(g_p.in_neighbors_offset);
	cudaFree(g_p.out_neighbors_offset);
	cudaFree(g_p.in_degrees);
	cudaFree(g_p.out_degrees);
}

void moveGraphToDevice(Graph& g, device_graph_pointers& g_p) {
	cudaMemcpy(g_p.in_neighbors, g.in_neighbors, g.E * sizeof(vertex), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.out_neighbors, g.out_neighbors, g.E * sizeof(vertex), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.in_neighbors_offset, g.in_neighbors_offset, (g.V+1) * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.out_neighbors_offset, g.out_neighbors_offset, (g.V+1) * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.in_degrees, g.in_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.out_degrees, g.out_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
}

void refreshGraphOnGPU(Graph& g, device_graph_pointers& g_p) {
	cudaMemcpy(g_p.in_degrees, g.in_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
	cudaMemcpy(g_p.out_degrees, g.out_degrees, g.V * sizeof(offset), cudaMemcpyHostToDevice);
}


// Accessory memory (used for peeling)
void allocateDeviceAccessoryMemory(Graph& g, device_accessory_pointers& a_p) {
	cudaMalloc(&(a_p.buffers), BUFFER_SIZE * BLOCK_NUMS * sizeof(vertex));
	cudaMalloc(&(a_p.bufferTails), BLOCK_NUMS * sizeof(unsigned));
	cudaMalloc(&(a_p.global_count), sizeof(unsigned));
	cudaMalloc(&(a_p.visited), g.V * sizeof(unsigned));	// unsigned instead of bool since atomicCAS only doesn't do bools
	cudaMalloc(&(a_p.core), g.V * sizeof(degree));
}

void deallocateDeviceAccessoryMemory(device_accessory_pointers& a_p) {
	cudaFree(a_p.buffers);
	cudaFree(a_p.bufferTails);
	cudaFree(a_p.global_count);
	cudaFree(a_p.visited);
	cudaFree(a_p.core);
}

// Maintenance memory (used for d-core maintenance)
void allocateDeviceMaintenanceMemory(Graph& g, device_maintenance_pointers& m_p) {
	cudaMalloc(&(m_p.k_max_max), sizeof(degree));
	cudaMalloc(&(m_p.m_value), sizeof(degree));
	cudaMalloc(&(m_p.k_max), g.V * sizeof(degree));
	cudaMalloc(&(m_p.l_max), g.V * sizeof(degree));
	cudaMalloc(&(m_p.new_l_max), g.V * sizeof(degree));
	cudaMalloc(&(m_p.compute), g.V * sizeof(unsigned));
	cudaMalloc(&(m_p.ED), g.V * sizeof(degree));
	cudaMalloc(&(m_p.PED), g.V * sizeof(degree));
	cudaMalloc(&(m_p.flag), sizeof(bool));
	cudaMalloc(&(m_p.histograms), (g.V + g.E) * sizeof(unsigned));
}

void deallocateDeviceMaintenanceMemory(device_maintenance_pointers& m_p) {
	cudaFree(m_p.k_max_max);
	cudaFree(m_p.m_value);
	cudaFree(m_p.k_max);
	cudaFree(m_p.l_max);
	cudaFree(m_p.new_l_max);
	cudaFree(m_p.compute);
	cudaFree(m_p.ED);
	cudaFree(m_p.PED);
	cudaFree(m_p.flag);
	cudaFree(m_p.histograms);
}

void putKmaxInDeviceMemory(device_maintenance_pointers& m_p, vector<degree>& kmax) {
	cudaMemcpy(m_p.k_max, kmax.data(), kmax.size() * sizeof(degree), cudaMemcpyHostToDevice);
}
void getKmaxFromDeviceMemory(device_maintenance_pointers& m_p, vector<degree>& kmax) {
	cudaMemcpy(kmax.data(), m_p.k_max, kmax.size() * sizeof(degree), cudaMemcpyDeviceToHost);
}
void putLmaxInDeviceMemory(device_maintenance_pointers& m_p, vector<degree>& lmax) {
	cudaMemcpy(m_p.l_max, lmax.data(), lmax.size() * sizeof(degree), cudaMemcpyHostToDevice);
}
void getLmaxFromDeviceMemory(device_maintenance_pointers& m_p, vector<degree>& lmax) {
	cudaMemcpy(lmax.data(), m_p.l_max, lmax.size() * sizeof(degree), cudaMemcpyDeviceToHost);
}

void initializeDeviceMaintenanceMemoryForKmax(unsigned V, device_maintenance_pointers& m_p) {
	cudaMemset(m_p.compute, 0, V * sizeof(unsigned));
	cudaMemset(m_p.ED, 0, V * sizeof(degree));
}

void initializeDeviceMaintenanceMemoryForKList(unsigned V, device_maintenance_pointers& m_p) {
	cudaMemset(m_p.compute, 0, V * sizeof(unsigned));
	cudaMemset(m_p.ED, 0, V * sizeof(degree));
}

void putModifiedEdgesInDeviceMemory(device_graph_pointers& g_p, vector<pair<vertex, vertex>> edges) {
	auto data = new vertex[edges.size() * 2];
	for (unsigned e = 0; e < edges.size(); ++e) {
		data[e*2] = edges[e].first;
		data[e*2+1] = edges[e].second;
	}
	cudaMemcpy(g_p.modified_edges, data, edges.size() * 2 * sizeof(vertex), cudaMemcpyHostToDevice);
}



#endif //D_CORE_CUDA_CUDAMEM_CUH