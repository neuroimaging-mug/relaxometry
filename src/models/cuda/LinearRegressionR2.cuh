/* 
 * File:   LinearRegressionR2.cuh
 * Author: christiantinauer
 *
 * Created on September 18, 2015, 11:10 AM
 */

#ifndef LINEARREGRESSIONR2_CUH
#define	LINEARREGRESSIONR2_CUH

#include "LinearRegressionT2.cuh"

template<typename NUMERICTYPE>
__device__ inline void linearRegressionR2(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	linearRegressionT2(input, output, length,
		parameters, parametersLength,
		constants, constantsLength);
	parameters[0] = 1 / parameters[0];
}

#endif	/* LINEARREGRESSIONR2_CUH */