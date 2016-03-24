/* 
 * File:   CudaCalculateAdjustedRsquareValue.cuh
 * Author: christiantinauer
 *
 * Created on September 27, 2015, 8:35 PM
 */

#ifndef CUDACALCULATEADJUSTEDRVALUE_CUH
#define	CUDACALCULATEADJUSTEDRVALUE_CUH

#include "../../../models/cuda/Statics.cuh"

template<typename NUMERICTYPE>
__device__	inline NUMERICTYPE CudaCalculateAdjustedRsquareValue(
	NUMERICTYPE* y_hat, int length, int parametersLength) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	NUMERICTYPE ySum = 0;
	NUMERICTYPE SSE = 0;
	for(int i = 0; i < length; ++i) {
		ySum += tex2D(floatTexture, i, index);
		NUMERICTYPE error = tex2D(floatTexture, i, index) - y_hat[i];
		SSE += error * error;
	}
	NUMERICTYPE yMean = ySum / (NUMERICTYPE)length;
	NUMERICTYPE SST = 0;
	for(int i = 0; i < length; ++i) {
		NUMERICTYPE diff = tex2D(floatTexture, i, index) - yMean;
		SST += diff * diff;
	}
	NUMERICTYPE adjustedRsquare = 1 - 
		(length - 1) / (length - parametersLength) *
		SSE / SST;
//	if(adjustedRsquare < 0)
//		adjustedRsquare = 0;
	return adjustedRsquare;
}

#endif	/* CUDACALCULATEADJUSTEDRVALUE_CUH */