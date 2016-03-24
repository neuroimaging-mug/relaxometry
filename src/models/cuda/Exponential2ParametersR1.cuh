/* 
 * File:   Exponential2ParamteresR1.cuh
 * Author: christiantinauer
 *
 * Created on January 18, 2015, 6:52 PM
 */

#ifndef EXPONENTIAL2PARAMETERSR1_CUH
#define	EXPONENTIAL2PARAMETERSR1_CUH

#include "Statics.cuh"

template<typename NUMERICTYPE>
__device__ inline void exponential2ParametersR1(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i = 0; i < length; ++i)
		output[i] = parameters[1] * (1 - tex2D(floatTexture, length + 1, index) * expf(-processConstants[13 + i] * parameters[0]));
}

#endif	/* EXPONENTIAL2PARAMETERSR1_CUH */