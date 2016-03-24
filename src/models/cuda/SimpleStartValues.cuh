/* 
 * File:   SimpleStartValues.cuh
 * Author: christiantinauer
 *
 * Created on September 18, 2015, 11:08 AM
 */

#ifndef SIMPLESTARTVALUES_CUH
#define	SIMPLESTARTVALUES_CUH

#include "Statics.cuh"

template<typename NUMERICTYPE>
__device__ inline void simpleStartValues(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	for(int index = 0; index < parametersLength; ++index)
		parameters[index] = processConstants[index];
}

#endif	/* SIMPLESTARTVALUES_CUH */