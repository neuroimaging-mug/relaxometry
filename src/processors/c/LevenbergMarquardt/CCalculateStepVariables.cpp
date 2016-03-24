/* 
 * File:   CCalculateStepVariables.cpp
 * Author: christiantinauer
 * 
 * Created on September 24, 2014, 5:22 PM
 */

#include "CCalculateStepVariables.hpp"

#include "../../../models/c/Model.hpp"
#include "CJacobian.hpp"

template<typename NUMERICTYPE>
void CCalculateStepVariables(
	short modelFunctionID,
	short residualWeightingFunctionID,
	short alphaWeightingFunctionID,
	NUMERICTYPE* x_values, NUMERICTYPE* y_values, NUMERICTYPE* residuals,
	NUMERICTYPE* weights, CStepVariables<NUMERICTYPE>* result, int length,
	NUMERICTYPE* parameters, int parameters_length,
	NUMERICTYPE* constants, int constants_length) {
	cModelFunction(modelFunctionID, x_values, result->y_hat, length,
		parameters, parameters_length,
		constants, constants_length);
	
	CJacobian(modelFunctionID, x_values, result->y_hat,
		residuals, result->dydp, length, 
		parameters, parameters_length,
		constants, constants_length);
	
	for(int i = 0; i < length; ++i)
		residuals[i] = (y_values[i] - result->y_hat[i]) *
			cWeightingFunction(residualWeightingFunctionID, weights, length, i); // residual error between model and data

	// alpha = dydp' * (dydp .* (weight_sq * ones(1, parameters_length)));
	for(int column = 0; column < parameters_length; ++column) {
		for(int row = 0; row < parameters_length; ++row) {
			result->alpha[column * parameters_length + row] = 0;
			for(int i = 0; i < length; ++i) {
				int left_index = i * parameters_length + column;
				NUMERICTYPE left = result->dydp[left_index];
				int right_index = i * parameters_length + row;
				NUMERICTYPE right = result->dydp[right_index];
				result->alpha[column * parameters_length + row] += 
					left * right * cWeightingFunction(alphaWeightingFunctionID, weights, length, i);
			}
		}
	}

	// beta  = dydp' * (weight_sq .* residuals);
	for(int p = 0; p < parameters_length; ++p) {
		NUMERICTYPE sum = 0;
		for(int i = 0; i < length; ++i) {
			int left_index = i * parameters_length + p;
			sum += result->dydp[left_index] * residuals[i];
		}
		result->beta[p] = sum;
	}

	// error = residuals' * (residuals .* weight_sq);
	result->error = 0;
	for(int i = 0; i < length; ++i)
		result->error += residuals[i] * residuals[i];
}

template
void CCalculateStepVariables(
	short modelFunctionID,
	short residualWeightingFunctionID,
	short alphaWeightingFunctionID,
	double* x_values, double* y_values, double* residuals,
	double* weights, CStepVariables<double>* result, int length,
	double* parameters, int parameters_length,
	double* constants, int constants_length);

template
void CCalculateStepVariables(
	short modelFunctionID,
	short residualWeightingFunctionID,
	short alphaWeightingFunctionID,
	float* x_values, float* y_values, float* residuals,
	float* weights, CStepVariables<float>* result, int length,
	float* parameters, int parameters_length,
	float* constants, int constants_length);