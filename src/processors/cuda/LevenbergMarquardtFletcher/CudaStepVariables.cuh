/* 
 * File:   CudaStepVariables.cuh
 * Author: christiantinauer
 *
 * Created on January 17, 2015, 6:17 PM
 */

#ifndef CUDASTEPVARIABLES_CUH
#define	CUDASTEPVARIABLES_CUH

#include "../../../models/cuda/Model.cuh"
#include "../../../models/cuda/Statics.cuh"
#include "CudaCalculateInverseMatrix.cuh"

#include "CudaJacobian.cu"

template<typename NUMERICTYPE>
struct CudaStepVariables {
	NUMERICTYPE* alpha;
	NUMERICTYPE* beta;
	NUMERICTYPE* y_hat;
	NUMERICTYPE error;
	NUMERICTYPE* dydp;
};

template<typename NUMERICTYPECALC>
__device__ bool CudaCheckStoppingCriteria(NUMERICTYPECALC* parameters, NUMERICTYPECALC* step,
	int parameters_length, NUMERICTYPECALC errorReduction, NUMERICTYPECALC* beta,
	int iteration, int max_iteration_count, NUMERICTYPECALC EPSILON_1, NUMERICTYPECALC EPSILON_2) {
//	const NUMERICTYPECALC EPSILON_2 = 1e-4; // convergence tolerance for parameters
//	const NUMERICTYPECALC EPSILON_3 = 1e-3; // convergence tolerance for Chi-square

//	NUMERICTYPECALC max = 0;
//	for(int i = 0; i < parameters_length; ++i) {
//		NUMERICTYPECALC value = abs(step[i] / parameters[i]);
//		if(value > max)
//			max = value;
//	}

//	if(max < EPSILON_2 && iteration > 2)
//		return true;

	if(abs(errorReduction) < EPSILON_2 && iteration > 2)
		return true;

	if(CudaMaxAbs(beta, parameters_length) < EPSILON_1 && iteration > 2)
		return true;

	if(iteration == max_iteration_count) {
		for(int i = 0; i < parameters_length; ++i)
			parameters[i] = 1; //NOT_CONVERGING;

		return true;
	}

	return false;
}

template
__device__ bool CudaCheckStoppingCriteria(float* parameters, float* step,
	int parameters_length, float errorReduction, float* beta,
	int iteration, int max_iteration_count, float EPSILON_1, float EPSILON_2);

template<typename NUMERICTYPE>
__device__ void CudaCalculateStepVariables(
	short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	NUMERICTYPE* x_values, NUMERICTYPE* residuals,
	NUMERICTYPE* weights, CudaStepVariables<NUMERICTYPE>* result, int length,
	NUMERICTYPE* parameters, int parameters_length) {
	cudaModelFunction(
		modelFunctionID,
		x_values, result->y_hat, length,
		parameters, parameters_length,
		(NUMERICTYPE*)NULL, 0);
	
	CudaJacobian(
		modelFunctionID,
		x_values, result->y_hat, residuals, result->dydp, length,
		parameters, parameters_length);
	
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	
	for(int i = 0; i < length; ++i)
		// residual error between model and data
		residuals[i] = (tex2D(floatTexture, i, index) - result->y_hat[i]) *
			cudaWeightingFunction(residualWeightingFunctionID, weights, length, i);
		
	// alpha = dydp' * (dydp .* (weight_sq * ones(1, parameters_length)));
	NUMERICTYPE currentAlpha;
	for(int column = 0; column < parameters_length; ++column) {
		for(int row = 0; row < parameters_length; ++row) {
			currentAlpha = 0;
			for(int i = 0; i < length; ++i) {
				int left_index = i * parameters_length + column;
				NUMERICTYPE left = result->dydp[left_index];
				int right_index = i * parameters_length + row;
				NUMERICTYPE right = result->dydp[right_index];
				currentAlpha += left * right *
					cudaWeightingFunction(alphaWeightingFunctionID, weights, length, i);
			}
			result->alpha[column * parameters_length + row] = currentAlpha;
		}
	}

	// beta  = dydp' * (weight_sq .* residuals);
	for(int p = 0; p < parameters_length; ++p) {
		NUMERICTYPE sum = 0;
		for(int i = 0; i < length; ++i) {
			int left_index = i * parameters_length + p;
			NUMERICTYPE right = residuals[i]; //weights?
			NUMERICTYPE left = result->dydp[left_index];
			sum += left * right;
		}
		result->beta[p] = sum;
	}

	// error = residuals' * (residuals .* weight_sq);
	result->error = 0;
	for(int i = 0; i < length; ++i)
		result->error += residuals[i] * residuals[i];
}

template
__device__ void CudaCalculateStepVariables(
	short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	float* x_values, float* residuals,
	float* weights, CudaStepVariables<float>* result, int length,
	float* parameters, int parameters_length);

template<typename NUMERICTYPECALC>
__device__ void CudaCalculateStep(NUMERICTYPECALC* step,
	CudaStepVariables<NUMERICTYPECALC>* step_variables, NUMERICTYPECALC lambda, int parametersCount) {
	//step = (alpha + lambda *diag (diag(alpha))) \ beta;
	int size = parametersCount * parametersCount;
	NUMERICTYPECALC matrix[9];
	for(int index = 0; index < size; ++index)
		matrix[index] = step_variables->alpha[index] * 
			(1 + (index % (parametersCount + 1) == 0 ? lambda : 0));
	
	NUMERICTYPECALC inverseMatrix[9];
	CudaCalculateInverseMatrix(matrix, inverseMatrix, parametersCount);
	
	for(int index = 0; index < parametersCount; ++index) {
		step[index] = 0;
		for(int innerIndex = 0; innerIndex < parametersCount; ++innerIndex)
			step[index] += inverseMatrix[index * parametersCount + innerIndex] * step_variables->beta[innerIndex];
	}
}

template
__device__ void CudaCalculateStep(float* step,
	CudaStepVariables<float>* step_variables, float lambda, int parametersCount);

template<typename NUMERICTYPECALC>
__device__ void CudaCalculateParametersTry(NUMERICTYPECALC* parameters_try, NUMERICTYPECALC* parameters,
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

template
__device__ void CudaCalculateParametersTry(float* parameters_try, float* parameters,
	float* step, float* parameters_min, float* parameters_max,
	int parameters_length);

template<typename NUMERICTYPE>
__device__ NUMERICTYPE CudaMaxAbs(NUMERICTYPE* values, int length) {
	NUMERICTYPE max = 0;
	for(int i = 0; i < length; ++i) {
		NUMERICTYPE value = abs(values[i]);
		if (value > max)
			max = value;
	}
	return max;
}

template
__device__ float CudaMaxAbs(float* values, int length);

#endif	/* CUDASTEPVARIABLES_CUH */