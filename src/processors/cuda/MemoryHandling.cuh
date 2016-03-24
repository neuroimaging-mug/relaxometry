#ifndef _MemoryHandling
#define _MemoryHandling

#include "ErrorHandling.cuh"

template<typename T> T* AllocOnDevice(int count) {
	T* result;
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&result, count * sizeof(T)));
	return result;
}

template<typename T> T* AllocAndCopyToDevice(T* data, int count) {
	T* gpuData;
	int size = count * sizeof (T);
	
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&gpuData, size));
	HANDLE_CUDA_ERROR(cudaMemcpy(gpuData, data, size, cudaMemcpyHostToDevice));
	
	return gpuData;
}

template<typename T> T* CopyFromDeviceAndFree(T* gpuData, int count) {
	int size = count * sizeof (T);

	T* result = (T*)malloc(size);
	HANDLE_CUDA_ERROR(cudaMemcpy(result, gpuData, size, cudaMemcpyDeviceToHost));

	cudaFree(gpuData);
	gpuData = NULL;

	return result;
}

template<class T> T* AllocPitchOnDevice(size_t *pitch, size_t width, size_t height) {
	T* result;
	HANDLE_CUDA_ERROR(cudaMallocPitch((void**)&result, pitch, width*sizeof(T), height));
	return result;
}


template<class T> T* AllocAndCopyPitchToDevice(const T* data, size_t *pitch, size_t width, size_t height, size_t spitch) {
	T* gpuData;
	HANDLE_CUDA_ERROR(cudaMallocPitch((void**)&gpuData, pitch, width*sizeof(T), height));
	HANDLE_CUDA_ERROR(cudaMemcpy2D(gpuData, *pitch, data, spitch*sizeof(T), width*sizeof(T), height, cudaMemcpyHostToDevice));
	*pitch = *pitch/sizeof(T);
	return gpuData;
}

#endif