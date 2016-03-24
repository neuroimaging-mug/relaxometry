#include "../../../models/cuda/Model.cuh"

template<typename NUMERICTYPE>
__device__ void CudaJacobian(
	short modelFunctionID,
	NUMERICTYPE* x_values, NUMERICTYPE* y_hat, NUMERICTYPE* y_forward,
	NUMERICTYPE* result, int length,
	NUMERICTYPE* parameters, int parameters_length) {
	const NUMERICTYPE dp = -0.01f;
		
	for(int j = 0; j < parameters_length; ++j) {
		NUMERICTYPE parameter = parameters[j];
		NUMERICTYPE delta = dp * (1 + parameter);
		parameters[j] = parameter + delta;

		if(abs(delta) > 1e-5f) {
			cudaModelFunction(
				modelFunctionID,
				x_values, y_forward, length,
				parameters, parameters_length, 
				(NUMERICTYPE*)NULL, 0);
			for(int i = 0; i < length; ++i) {
				int index = j + parameters_length * i;
				result[index] = (y_forward[i] - y_hat[i]) / delta;
			}
		}
		// else
		//	printf("WARNING: delta was too small: %f <= %f\n", abs(delta), 1e-5f);

		parameters[j] = parameter;
	}
}

template
__device__ void CudaJacobian(
	short modelFunctionID,
	float* x_values, float* y_hat, float* y_forward,
	float* result, int length,
	float* parameters, int parameters_length);