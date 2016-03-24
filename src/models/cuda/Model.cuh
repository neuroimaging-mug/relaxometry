/* 
 * File:   Models.cuh
 * Author: christiantinauer
 *
 * Created on August 17, 2015, 5:15 PM
 */

#ifndef MODEL_CUH
#define	MODEL_CUH

#include "Exponential2ParametersR1.cuh"
#include "Exponential2ParametersT1.cuh"
#include "Exponential3ParametersR1.cuh"
#include "Exponential3ParametersT1.cuh"
#include "ExponentialR2.cuh"
#include "ExponentialT2.cuh"
#include "LukzenSavelov.cuh"
#include "LinearRegressionR2.cuh"
#include "LinearRegressionT2.cuh"
#include "SimpleStartValues.cuh"

#include "LinearWeighting.cuh"
#include "InverseMinimumWeighting.cuh"
#include "InverseQuadraticWeighting.cuh"

template<typename NUMERICTYPE>
__device__ inline void cudaModelFunction(
	short modelFunctionID,
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	switch(modelFunctionID) {
		case 0:
			exponential2ParametersR1(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 1:
			exponential2ParametersT1(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 2:
			exponential3ParametersR1(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 3:
			exponential3ParametersT1(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 4: 
			exponentialR2(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 5:
			exponentialT2(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 6:
			lukzenSavelov(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 7:
			linearRegressionR2(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 8: 
			linearRegressionT2(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 9:
			simpleStartValues(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
	}
}

template<typename NUMERICTYPE>
__device__ inline NUMERICTYPE cudaWeightingFunction(
	short weightingFunctionID, NUMERICTYPE* weights, int length, int index) {
	switch(weightingFunctionID) {
		case 0: return linearWeighting(weights, length, index);
		case 1: return inverseMinimumWeighting(weights, length, index);
		case 2: return inverseQuadraticWeighting(weights, length, index);
		default: return 1;
	}
}

#endif	/* MODEL_CUH */