/* 
 * File:   Exponential3ParamteresR1.cuh
 * Author: christiantinauer
 *
 * Created on January 18, 2015, 6:52 PM
 */

#ifndef EXPONENTIAL3PARAMETERSR1_CUH
#define	EXPONENTIAL3PARAMETERSR1_CUH

template<typename NUMERICTYPE>
__device__ inline void exponential3ParametersR1(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	for(int i = 0; i < length; ++i)
		output[i] = parameters[1] * (1 - parameters[2] * expf(-processConstants[13 + i] * parameters[0]));
}

#endif	/* EXPONENTIAL3PARAMETERSR1_CUH */