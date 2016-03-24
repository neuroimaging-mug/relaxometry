#ifndef _ErrorHandling
#define _ErrorHandling

#include "../../includes/StringExtensions.h"
#include <string>
#include <stdexcept>
#include <stdio.h>
#include <driver_types.h>
#include <cuda_runtime.h>

using namespace std;

//to suppress annoying warning about defined and unused function
#ifdef __GNUC__
#define FUNCTION_IS_NOT_USED __attribute__ ((unused))
#else
#define FUNCTION_IS_NOT_USED
#endif

static void FUNCTION_IS_NOT_USED HandleCudaError(cudaError_t error, const char *file, int line) {
	if(error != cudaSuccess)
		throw runtime_error(string(cudaGetErrorString(error)) + " in " + string(file) + " at line " + to_string(line));
}

#define HANDLE_CUDA_ERROR(error) (HandleCudaError(error, __FILE__, __LINE__))

#endif