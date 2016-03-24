/* 
 * File:   Model.hpp
 * Author: christiantinauer
 *
 * Created on October 1, 2015, 9:22 PM
 */

#ifndef MODEL_HPP
#define	MODEL_HPP

#include "ModelService.hpp"
#include "../../includes/StringExtensions.h"

#include <stdexcept>

template<typename NTCALC>
inline void cModelFunction(
	short modelFunctionID,
	NTCALC* input, NTCALC* output, int length,
	NTCALC* parameters, int parametersLength,
	NTCALC* constants, int constantsLength) {
	switch(modelFunctionID) {
		case 0:
			ModelService::Exponential2ParametersR1(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 1:
			ModelService::Exponential2ParametersT1(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 2:
			ModelService::Exponential3ParametersR1(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 3:
			ModelService::Exponential3ParametersT1(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 4: 
			ModelService::ExponentialR2(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 5:
			ModelService::ExponentialT2(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 6:
			ModelService::LukzenSavelov(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 7:
			ModelService::LinearRegressionR2(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 8: 
			ModelService::LinearRegressionT2(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		case 9:
			ModelService::SimpleStartValues(input, output, length, 
				parameters, parametersLength,
				constants, constantsLength);
			break;
		default:
			throw runtime_error("ModelFunctionID does not exist. ID: " + to_string(modelFunctionID) + ".");
	}
}

template<typename NTCALC>
inline NTCALC cWeightingFunction(
	short weightingFunctionID, NTCALC* weights, int length, int index) {
	switch(weightingFunctionID) {
		case 0: return ModelService::LinearWeighting(weights, length, index);
		case 1: return ModelService::InverseMinimumWeighting(weights, length, index);
		case 2: return ModelService::InverseQuadraticWeighting(weights, length, index);
		default: return 1;
	}
}

#endif	/* MODEL_HPP */