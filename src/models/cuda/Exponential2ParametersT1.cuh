/* 
 * File:   Exponential2ParamteresT1.cuh
 * Author: christiantinauer
 *
 * Created on January 18, 2015, 6:52 PM
 */

#ifndef EXPONENTIAL2PARAMETERST1_CUH
#define	EXPONENTIAL2PARAMETERST1_CUH

#include "Statics.cuh"

template<typename NUMERICTYPE>
__device__ inline void exponential2ParametersT1(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i = 0; i < length; ++i)
		output[i] = parameters[1] * (1 - tex2D(floatTexture, length + 1, index) * expf(-processConstants[13 + i] / parameters[0]));
}

#endif	/* EXPONENTIAL2PARAMETERST1_CUH */