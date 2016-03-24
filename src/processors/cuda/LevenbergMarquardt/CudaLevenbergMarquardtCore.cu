#include "../../../models/cuda/Model.cuh"
#include "../../../models/cuda/Statics.cuh"

#include "../LevenbergMarquardtFletcher/CudaStepVariables.cuh"

template<typename NUMERICTYPECALC>
__device__ NUMERICTYPECALC* CudaLevenbergMarquardtCore(
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
//	const NUMERICTYPECALC EPSILON_1 = 1e-6; // convergence tolerance for gradient
	const NUMERICTYPECALC EPSILON_4 = 1e-2; // determines acceptance of a L-M step
	const NUMERICTYPECALC LAMBDA_0 = 1e-2; // initial value of damping parameter, lambda
	const NUMERICTYPECALC LAMBDA_UP = 11; // factor for increasing lambda
	const NUMERICTYPECALC LAMBDA_DOWN = 9; // factor for decreasing lambda

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
	NUMERICTYPECALC error_old = step_variables->error;
	NUMERICTYPECALC error = step_variables->error;
	NUMERICTYPECALC errorReduction = step_variables->error;

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
			CudaCalculateStepVariables(
				modelFunctionID,
				residualWeightingFunctionID, alphaWeightingFunctionID,
				x_values, delta_y_memory,
				weights, step_variables, echoesCount,
				parameters, parameters_length);

			// decrease lambda ==> Gauss-Newton method
			lambda = max((NUMERICTYPECALC)(lambda / LAMBDA_DOWN), (NUMERICTYPECALC)1.e-7);
		} else { //it is not better
			error = error_old; // do not accept a_try

			// increase lambda  ==> gradient descent method
			lambda = min((NUMERICTYPECALC)(lambda * LAMBDA_UP), (NUMERICTYPECALC)1.e7);
		}
		
		stop = CudaCheckStoppingCriteria(
			parameters, step, parameters_length,
			errorReduction, step_variables->beta, iteration, max_iteration_count,
			EPSILON_1, EPSILON_2);
		++iteration;
	}
	
	return parameters;
}