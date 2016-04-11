#include "../../ProcessorService.hpp"
#include "../../../models/c/Model.hpp"
#include "CCalculateStepVariables.hpp"

#include <algorithm>

using namespace std;

template<typename NUMERICTYPECALC>
bool CCheckStoppingCriteria(NUMERICTYPECALC* parameters, NUMERICTYPECALC* step,
	int parameters_length, NUMERICTYPECALC errorReduction, NUMERICTYPECALC* beta,
	int iteration, int max_iteration_count, NUMERICTYPECALC EPSILON_1,
	NUMERICTYPECALC EPSILON_2) {
//	const NUMERICTYPECALC EPSILON_2 = 1e-4; // convergence tolerance for parameters
//	const NUMERICTYPECALC EPSILON_3 = 1e-3; // convergence tolerance for Chi-square
//
//	NUMERICTYPECALC max = 0;
//	for(int i = 0; i < parameters_length; ++i) {
//		NUMERICTYPECALC value = abs(step[i] / parameters[i]);
//		if(value > max)
//			max = value;
//	}
//
//	if(max < EPSILON_2 && iteration > 2)
//		return true;

	if(abs(errorReduction) < EPSILON_2 && iteration > 2)
		return true;

	if(ProcessorService::MaxAbs(beta, parameters_length) < EPSILON_1 && iteration > 2)
		return true;

	if(iteration == max_iteration_count) {
		//for(int i = 0; i < parameters_length; ++i)
		//	parameters[i] = 1; //NOT_CONVERGING;

		return true;
	}

	return false;
}

template<typename NTCALC>
void CCalculateInverseMatrix(NTCALC* matrix, NTCALC* inverseMatrix, int parametersCount) {
	if(parametersCount == 2) {
		NTCALC det = 1 / (matrix[0] * matrix[3] - matrix[1] * matrix[2]);
		inverseMatrix[0] = matrix[3] * det;
		inverseMatrix[1] = -matrix[1] * det;
		inverseMatrix[2] = -matrix[2] * det;
		inverseMatrix[3] = matrix[0] * det;
	} else if(parametersCount == 3) {
		NTCALC det = 1 / (
			(	matrix[0] * matrix[4] * matrix[8] +
				matrix[1] * matrix[5] * matrix[6] +
				matrix[2] * matrix[3] * matrix[7]) -
			(	matrix[2] * matrix[4] * matrix[6] +
				matrix[0] * matrix[5] * matrix[7] +
				matrix[1] * matrix[3] * matrix[8]));
		inverseMatrix[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * det;
		inverseMatrix[1] = -(matrix[1] * matrix[8] - matrix[2] * matrix[7]) * det;
		inverseMatrix[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * det;
		inverseMatrix[3] = -(matrix[3] * matrix[8] - matrix[5] * matrix[6]) * det;
		inverseMatrix[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * det;
		inverseMatrix[5] = -(matrix[0] * matrix[5] - matrix[2] * matrix[3]) * det;
		inverseMatrix[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * det;
		inverseMatrix[7] = -(matrix[0] * matrix[7] - matrix[1] * matrix[6]) * det;
		inverseMatrix[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * det;
	}
}

template<typename NUMERICTYPECALC>
void CCalculateStep(NUMERICTYPECALC* step,
	CStepVariables<NUMERICTYPECALC>* step_variables, NUMERICTYPECALC lambda, int parametersCount) {
	//step = (alpha + lambda *diag (diag(alpha))) \ beta;
	int size = parametersCount * parametersCount;
	NUMERICTYPECALC matrix[size];
	for(int index = 0; index < size; ++index)
		matrix[index] = step_variables->alpha[index] * 
			(1 + (index % (parametersCount + 1) == 0 ? lambda : 0));
	
	NUMERICTYPECALC inverseMatrix[size];
	CCalculateInverseMatrix(matrix, inverseMatrix, parametersCount);
	
	for(int index = 0; index < parametersCount; ++index) {
		step[index] = 0;
		for(int innerIndex = 0; innerIndex < parametersCount; ++innerIndex)
			step[index] += inverseMatrix[index * parametersCount + innerIndex] * step_variables->beta[innerIndex]; 
	}
}

template<typename NUMERICTYPECALC>
void CCalculateParametersTry(NUMERICTYPECALC* parameters_try, NUMERICTYPECALC* parameters,
	NUMERICTYPECALC* step, NUMERICTYPECALC* parameters_min, NUMERICTYPECALC* parameters_max,
	int parameters_length) {
	//p_try = p + step(idx);
	//p_try = min(max(p_min ,p_try), p_max);           % apply constraints
	for(int i = 0; i < parameters_length; ++i) {
		NUMERICTYPECALC p_try = parameters[i] + step[i];
		if(p_try > parameters_max[i])
			p_try = parameters_max[i];
		else if(p_try < parameters_min[i])
			p_try = parameters_min[i];

		parameters_try[i] = p_try;
	}
}

template<typename NUMERICTYPECALC>
NUMERICTYPECALC* CLevenbergMarquardtCore(
	short modelFunctionID,
	short residualWeightingFunctionID,
	short alphaWeightingFunctionID,
	NUMERICTYPECALC* x_values, NUMERICTYPECALC* y_values, int length, NUMERICTYPECALC* delta_y_memory,
  NUMERICTYPECALC* parameters, int parameters_length,
  NUMERICTYPECALC* constants, int constants_length,
  NUMERICTYPECALC* weights, NUMERICTYPECALC* parameters_min, NUMERICTYPECALC* parameters_max,
  NUMERICTYPECALC* step, NUMERICTYPECALC* parameters_try, CStepVariables<NUMERICTYPECALC>* step_variables,
	NUMERICTYPECALC EPSILON_1, NUMERICTYPECALC EPSILON_2) {
//	const NUMERICTYPECALC EPSILON_1 = 1e-6; // convergence tolerance for gradient
	const NUMERICTYPECALC EPSILON_4 = 1e-2; // determines acceptance of a L-M step
	const NUMERICTYPECALC LAMBDA_0 = 1e-2; // initial value of damping parameter, lambda
	const NUMERICTYPECALC LAMBDA_UP = 11; // factor for increasing lambda
	const NUMERICTYPECALC LAMBDA_DOWN = 9; // factor for decreasing lambda

	bool stop = false;
	
	CCalculateStepVariables(
		modelFunctionID,
		residualWeightingFunctionID,
		alphaWeightingFunctionID,
		x_values, y_values, delta_y_memory,
		weights, step_variables, length, 
		parameters, parameters_length,
		constants, constants_length);
	
	if(ProcessorService::MaxAbs(step_variables->beta, parameters_length) < EPSILON_1)
		stop = true;

	NUMERICTYPECALC lambda = LAMBDA_0;
	NUMERICTYPECALC error_old = step_variables->error;
	NUMERICTYPECALC error = step_variables->error;
	NUMERICTYPECALC errorReduction = step_variables->error;

	int max_iteration_count = 20 * parameters_length;
	int iteration = 1;
	
	while(!stop && iteration < max_iteration_count) {
		CCalculateStep(step, step_variables, lambda, parameters_length);
		CCalculateParametersTry(parameters_try, parameters, step,
			parameters_min, parameters_max, parameters_length);

		//delta_y = y_dat - feval(func,t,p_try,c);       % residual error using p_try
		//error_try = delta_y' * ( delta_y .* weight_sq );  % Chi-squared error criteria
		NUMERICTYPECALC error_try = 0;
		NUMERICTYPECALC* y_try = delta_y_memory;
		cModelFunction(modelFunctionID,
			x_values, y_try, length, 
			parameters_try, parameters_length,
			constants, constants_length);
		for(int i = 0; i < length; ++i) {
			NUMERICTYPECALC delta_y = (y_values[i] - y_try[i]) * 
				cWeightingFunction(residualWeightingFunctionID, weights, length, i);
			error_try += delta_y * delta_y;
		}

		//rho = (error - error_try) / ( 2*step' * (lambda * step + beta) ); % Nielsen
		NUMERICTYPECALC right = 0;
		for(int i = 0; i < parameters_length; ++i)
			right += 2 * step[i] * (lambda * step[i] + step_variables->beta[i]);
		NUMERICTYPECALC rho = (error - error_try) / right;

		if(rho > EPSILON_4) { // it is significantly better
			errorReduction = error_old - error;
			error_old = error;

			//p = p_try(:);			% accept p_try
			for(int i = 0; i < parameters_length; ++i)
				parameters[i] = parameters_try[i];
			
			//[alpha,beta,error,y_hat,dydp] = lm_matx(func,t,p,y_dat,weight_sq,dp,c);
			CCalculateStepVariables(
				modelFunctionID,
				residualWeightingFunctionID,
				alphaWeightingFunctionID,
				x_values, y_values, delta_y_memory,
				weights, step_variables, length,
				parameters, parameters_length,
				constants, constants_length);

			// decrease lambda ==> Gauss-Newton method
			lambda = max((NUMERICTYPECALC)(lambda / LAMBDA_DOWN), (NUMERICTYPECALC)1.e-7);
		} else { //it is not better
			error = error_old; // do not accept a_try

			// increase lambda  ==> gradient descent method
			lambda = min((NUMERICTYPECALC)(lambda * LAMBDA_UP), (NUMERICTYPECALC)1.e7);
		}
		
		stop = CCheckStoppingCriteria(
			parameters, step, parameters_length,
			errorReduction, step_variables->beta, iteration, max_iteration_count,
			EPSILON_1, EPSILON_2);
		++iteration;
	}
	
	return parameters;
}