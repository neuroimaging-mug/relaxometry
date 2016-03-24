/* 
 * File:   Jacobian.cpp
 * Author: christiantinauer
 * 
 * Created on September 24, 2014, 3:45 PM
 */

#include "CJacobian.hpp"

#include "../../../models/c/Model.hpp"

#include <cmath>
#include <stdio.h>

using namespace std;

template<typename NUMERICTYPE>
void CJacobian(
	short modelFunctionID,
	NUMERICTYPE* x_values, NUMERICTYPE* y_values, NUMERICTYPE* y_forward,
	NUMERICTYPE* result, int length,
	NUMERICTYPE* parameters, int parameters_length,
	NUMERICTYPE* constants, int constants_length) {
	const NUMERICTYPE dp = -0.01;
	
	for(int j = 0; j < parameters_length; ++j) {
		NUMERICTYPE parameter = parameters[j];
		NUMERICTYPE delta = dp * (1 + parameter);
		parameters[j] = parameter + delta;

		if(abs(delta) > 1e-5f) {
			cModelFunction(modelFunctionID, x_values, y_forward, length,
				parameters, parameters_length,
				constants, constants_length);
			for(int i = 0; i < length; ++i) {
				int index = j + parameters_length * i;
				result[index] = (y_forward[i] - y_values[i]) / delta;
			}
		} else
			printf("WARNING: delta was too small: %f <= %f\n", abs(delta), 1e-5f);

		parameters[j] = parameter;
	}
}

template
void CJacobian(
	short modelFunctionID,
	double* x_values, double* y_values, double* y_forward,
	double* result, int length,
	double* parameters, int parameters_length,
	double* constants, int constants_length);

template
void CJacobian(
	short modelFunctionID,
	float* x_values, float* y_values, float* y_forward,
	float* result, int length,
	float* parameters, int parameters_length,
	float* constants, int constants_length);