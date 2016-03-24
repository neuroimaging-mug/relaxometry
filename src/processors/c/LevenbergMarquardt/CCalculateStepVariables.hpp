/* 
 * File:   CCalculateStepVariables.hpp
 * Author: christiantinauer
 *
 * Created on September 24, 2014, 5:22 PM
 */

#ifndef CCALCULATESTEPVARIABLES_HPP
#define	CCALCULATESTEPVARIABLES_HPP

template<typename NUMERICTYPE>
struct CStepVariables {
	NUMERICTYPE* alpha;
	NUMERICTYPE* beta;
	NUMERICTYPE* y_hat;
	NUMERICTYPE error;
	NUMERICTYPE* dydp;
};

template<typename NUMERICTYPE>
void CCalculateStepVariables(
	short modelFunctionID,
	short residualWeightingFunctionID,
	short alphaWeightingFunctionID,
	NUMERICTYPE* x_values, NUMERICTYPE* y_values, NUMERICTYPE* residuals,
	NUMERICTYPE* weights, CStepVariables<NUMERICTYPE>* result, int length,
	NUMERICTYPE* parameters, int parameters_length,
	NUMERICTYPE* constants, int constants_length);

#endif	/* CCALCULATESTEPVARIABLES_HPP */