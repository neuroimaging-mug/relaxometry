/* 
 * File:   ExponentialT2.cuh
 * Author: christiantinauer
 *
 * Created on January 18, 2015, 6:52 PM
 */

#ifndef EXPONENTIALT2_CUH
#define	EXPONENTIALT2_CUH

template<typename NUMERICTYPE>
__device__ inline void exponentialT2(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	for(int i = 0; i < length; ++i)
		output[i] = parameters[1] * expf(-processConstants[13 + i] / parameters[0]);
}

#endif	/* EXPONENTIALT2_CUH */