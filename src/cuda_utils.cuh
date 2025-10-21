#ifndef D_CORE_CUDA_CUDA_UTILS_H
#define D_CORE_CUDA_CUDA_UTILS_H
#include "common.h"

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


#endif //D_CORE_CUDA_CUDA_UTILS_H