/* 
 * File:   LinearRegressionT2.cuh
 * Author: christiantinauer
 *
 * Created on September 18, 2015, 11:08 AM
 */

#ifndef LINEARREGRESSIONT2_CUH
#define	LINEARREGRESSIONT2_CUH

#include "Statics.cuh"

template<typename NUMERICTYPE>
__device__ inline void linearRegressionT2(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	int offsetInImage = blockIdx.x * blockDim.x + threadIdx.x;
	
	float ySum = 0;
	for(int index = 0; index < length; ++index)
		ySum += tex2D(floatTexture, index, offsetInImage);
	float yMean = logf((float)(ySum / length));
	
	float meanTime = (processConstants[13] + processConstants[13 + length - 1]) / 2;
	float covariance = 0; // SS_xy
	float variance = 0; // SS_xx
	for(int index = 0; index < length; ++index) {
		float y = tex2D(floatTexture, index, offsetInImage);
		float yis = logf(y);
		float xis = processConstants[13 + index];
		float xDiff = xis - meanTime;
		covariance += xDiff * (yis - yMean);
		variance += xDiff * xDiff;
	}

	float k = covariance / variance;
	
	//t2
	float t2 = k == 0 ? 0 : 1 / -k;
	parameters[0] = t2;
	
	//m0
	float d = yMean - k * meanTime;
	float m0 = expf(d);
	parameters[1] = (NUMERICTYPE)m0;
}

#endif	/* LINEARREGRESSIONT2_CUH */