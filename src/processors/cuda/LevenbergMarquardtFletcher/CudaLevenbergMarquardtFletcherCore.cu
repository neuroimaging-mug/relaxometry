#include "../../../models/cuda/Model.cuh"
#include "../../../models/cuda/Statics.cuh"

#include "CudaCalculateInverseMatrix.cuh"
#include "CudaStepVariables.cuh"

template<typename NUMERICTYPECALC>
__device__ NUMERICTYPECALC CudaCalculateR(NUMERICTYPECALC actualReduction, NUMERICTYPECALC* step, CudaStepVariables<NUMERICTYPECALC>* step_variables, int parametersCount) {
	NUMERICTYPECALC predictedReduction = 0;
	if(parametersCount == 2)
		predictedReduction =
			-2 * (step[0] * step_variables->beta[0] + step[1] * step_variables->beta[1]) -
			(	step[0] * (step[0] * step_variables->alpha[0] + step[1] * step_variables->alpha[1]) +
				step[1] * (step[0] * step_variables->alpha[2] + step[1] * step_variables->alpha[3]));
	else if(parametersCount == 3)
		predictedReduction =
			-2 * (step[0] * step_variables->beta[0] + step[1] * step_variables->beta[1] + step[2] * step_variables->beta[2]) -
			(	step[0] * (step[0] * step_variables->alpha[0] + step[1] * step_variables->alpha[1] + step[2] * step_variables->alpha[2]) +
				step[1] * (step[0] * step_variables->alpha[3] + step[1] * step_variables->alpha[4] + step[2] * step_variables->alpha[5]) +
				step[2] * (step[0] * step_variables->alpha[6] + step[1] * step_variables->alpha[7] + step[2] * step_variables->alpha[8]));
	return actualReduction / predictedReduction;
}

template<typename NUMERICTYPECALC>
__device__ NUMERICTYPECALC CudaCalculateAlpha(NUMERICTYPECALC error_try, NUMERICTYPECALC error, NUMERICTYPECALC* step, CudaStepVariables<NUMERICTYPECALC>* step_variables, int parametersCount) {
	NUMERICTYPECALC divisor = 0;
	for(int index = 0; index < parametersCount; ++index)
		divisor += step[index] * step_variables->beta[index];
	return 1 / (2 - (error_try - error) / divisor);
}

template<typename NUMERICTYPECALC>
__device__ NUMERICTYPECALC CudaCalculateLambdaCutOff(CudaStepVariables<NUMERICTYPECALC>* step_variables, int parametersCount) {
	int size = parametersCount * parametersCount;
	NUMERICTYPECALC inverseMatrix[9];
	CudaCalculateInverseMatrix(step_variables->alpha, inverseMatrix, parametersCount);
	
	NUMERICTYPECALC sum = 0;
	for(int index = 0; index < size; ++index)
		sum += inverseMatrix[index] * inverseMatrix[index];
	
	return 1 / sqrt(sum);
}

template<typename NUMERICTYPECALC>
__device__ NUMERICTYPECALC* CudaLevenbergMarquardtFletcherCore(
	short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int columnCount, int rowCount, int echoesCount,
	NUMERICTYPECALC* weights, NUMERICTYPECALC* x_values,
	NUMERICTYPECALC* delta_y_memory,
  NUMERICTYPECALC* parameters, int parameters_length,
  NUMERICTYPECALC* parameters_min, NUMERICTYPECALC* parameters_max,
  NUMERICTYPECALC* step, NUMERICTYPECALC* parameters_try,
	CudaStepVariables<NUMERICTYPECALC>* step_variables,
	NUMERICTYPECALC EPSILON_1, NUMERICTYPECALC EPSILON_2) {
	const NUMERICTYPECALC LAMBDA_0 = 0; // initial value of damping parameter, lambda
	
	bool stop = false;
	
	CudaCalculateStepVariables(
		modelFunctionID,
		residualWeightingFunctionID, alphaWeightingFunctionID,
		x_values, delta_y_memory,
		weights, step_variables, echoesCount,
		parameters, parameters_length);
	
	if(CudaMaxAbs(step_variables->beta, parameters_length) < EPSILON_1)
		stop = true;

	NUMERICTYPECALC lambda = LAMBDA_0;
	NUMERICTYPECALC error = step_variables->error;
	NUMERICTYPECALC errorReduction = error;

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	
	int max_iteration_count = 20 * parameters_length;
	int iteration = 1;
	
	while(!stop && iteration < max_iteration_count) {
		CudaCalculateStep(step, step_variables, lambda, parameters_length);
		CudaCalculateParametersTry(parameters_try, parameters, step,
			parameters_min, parameters_max, parameters_length);

		//delta_y = y_dat - feval(func,t,p_try,c);       % residual error using p_try
		//error_try = delta_y' * ( delta_y .* weight_sq );  % Chi-squared error criteria
		NUMERICTYPECALC error_try = 0;
		NUMERICTYPECALC* y_try = delta_y_memory;
		cudaModelFunction(modelFunctionID, x_values, y_try, echoesCount,
			parameters_try, parameters_length, 
			(NUMERICTYPECALC*)NULL, 0);
		for(int i = 0; i < echoesCount; i++) {
			NUMERICTYPECALC delta_y = (tex2D(floatTexture, i, index) - y_try[i]) *
				cudaWeightingFunction(residualWeightingFunctionID, weights, echoesCount, i);
			error_try += delta_y * delta_y;
		}
		
		NUMERICTYPECALC actualreduction = error - error_try;
		NUMERICTYPECALC R = CudaCalculateR(actualreduction, step, step_variables, parameters_length);
		if(R < 0.25) { //rho
			NUMERICTYPECALC alpha = CudaCalculateAlpha(error_try, error, step, step_variables, parameters_length);
			NUMERICTYPECALC nu = 1.0 / alpha;
			if(nu < 2.0)
				nu = 2.0;
			else if(nu > 10.0)
				nu = 10.0;
			
			if(lambda == 0.0) {
				lambda = CudaCalculateLambdaCutOff(step_variables, parameters_length);
				nu /= 2.0; 
			}
			
			lambda *= nu;
		} else if(R > 0.75) { //sigma
			lambda /= 2.0;
			
			if(lambda < CudaCalculateLambdaCutOff(step_variables, parameters_length))
				lambda = 0.0;
		}
		
		if(error_try < error) {
			//set new params
			for(int i = 0; i < parameters_length; ++i)
				parameters[i] = parameters_try[i];
		
			errorReduction = error - error_try;
			error = error_try;
		
			CudaCalculateStepVariables(
				modelFunctionID,
				residualWeightingFunctionID, alphaWeightingFunctionID,
				x_values, delta_y_memory,
				weights, step_variables, echoesCount,
				parameters, parameters_length);
		}
		
		stop = CudaCheckStoppingCriteria(
			parameters, step, parameters_length,
			errorReduction, step_variables->beta, iteration, max_iteration_count, 
			EPSILON_1, EPSILON_2);
		++iteration;
	}
	
	return parameters;
}