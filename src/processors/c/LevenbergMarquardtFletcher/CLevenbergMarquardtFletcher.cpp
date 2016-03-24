#include "../../ProcessorService.hpp"
#include "../../../models/Model.hpp"
#include "../LevenbergMarquardt/CCalculateStepVariables.hpp"

#include <limits>

using namespace std;

#include "CLevenbergMarquardtFletcherCore.cpp"

template<typename NTINPUT, typename NTOUTPUT, typename NTCALC>
void CLevenbergMarquardtFletcher(
	short startModelFunctionID,
	short modelFunctionID,
	short residualWeightingFunctionID,
	short alphaWeightingFunctionID,
	int parametersCount,
	int offsetInImage, int echoesCount, int sliceSize, int sliceCount, 
	NTCALC* x_values, NTINPUT* signal, NTCALC threshold,
	NTCALC* constants, int constantsLength,
	NTCALC* parameterBoundaries,
	NTOUTPUT* result, NTCALC* weights, NTCALC* all_y_values, NTCALC* tempCalculateError,
  NTCALC* y_hat, NTCALC* dydp, NTCALC EPSILON_1, NTCALC EPSILON_2) {
	int offsetInSignal = offsetInImage;
	NTINPUT value = signal[offsetInSignal];
	if(!isnan(value) && value <= (NTINPUT)threshold)
		return;
	
	NTCALC parameters_min[parametersCount];
	NTCALC parameters_max[parametersCount];
	for(int index = 0; index < parametersCount; ++index) {
		parameters_min[index] = parameterBoundaries[index * 2];
		parameters_max[index] = parameterBoundaries[index * 2 + 1];
	}

	NTCALC x_values_valid[echoesCount];
	NTCALC* y_values = all_y_values + offsetInSignal;
	int validEchoesCount = 0;
	for(int echoIndex = 0; echoIndex < echoesCount; ++echoIndex) {
		NTCALC y_value = signal[offsetInSignal + echoIndex * sliceCount * sliceSize];
		if(!isnan(y_value)) {
			y_values[validEchoesCount] = y_value;
			x_values_valid[validEchoesCount++] = x_values[echoIndex];
		}
	}
	
	bool calc = validEchoesCount > parametersCount;
	
	//Start values
	NTCALC parameters[parametersCount];
	if(calc)
		cModelFunction(startModelFunctionID,
			x_values_valid, y_values, validEchoesCount,
			parameters, parametersCount, constants, constantsLength);

	NTCALC delta_p[parametersCount];
	NTCALC parameters_try[parametersCount];

	NTCALC alpha[parametersCount * parametersCount];
	NTCALC beta[parametersCount];
	CStepVariables<NTCALC> stepVariables;
	stepVariables.alpha = alpha;
	stepVariables.beta = beta;
	stepVariables.dydp = dydp + offsetInImage * echoesCount * parametersCount;
	stepVariables.y_hat = y_hat + offsetInImage * echoesCount;

	if(calc)
		CLevenbergMarquardtFletcherCore(
			modelFunctionID,
			residualWeightingFunctionID,
			alphaWeightingFunctionID,
			x_values_valid, y_values, validEchoesCount,
			tempCalculateError + offsetInImage * echoesCount,
			parameters, parametersCount, constants, constantsLength, weights,
			parameters_min, parameters_max, delta_p, parameters_try, &stepVariables, 
			EPSILON_1, EPSILON_2);
	
	//p1 (T1/T2)
	result[offsetInImage] = (NTOUTPUT)parameters[0];
	
	//p2 (M0)
	result[offsetInImage + sliceCount * sliceSize] = (NTOUTPUT)parameters[1];
	
	//p3 (FA)
	if(parametersCount == 3)
		result[offsetInImage + sliceCount * sliceSize * 2] = (NTOUTPUT)parameters[2];
		
	//GOF
	if(calc) {
		//check for not converging
		calc = false;
		for(int index = 0; index < parametersCount; ++index)
			if(parameters[index] != 1) {
				calc = true;
				break;
			}
	}
	
	result[offsetInImage + sliceCount * sliceSize * parametersCount] = calc
		? ProcessorService::CalculateAdjustedRsquareValue(y_values, 
				stepVariables.y_hat, validEchoesCount, parametersCount)
		: 0;
}

template<typename NTINPUT, typename NTOUTPUT, typename NTCALC>
NTOUTPUT* CProcessLevenbergMarquardtFletcher(
	short startModelFunctionID,
	short modelFunctionID,
	short residualWeightingFunctionID,
	short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount,
	int sliceCount, NTCALC* echotimes, NTCALC* weights,
	NTINPUT* data, NTCALC threshold,
	NTCALC* constants, int constantsLength, NTCALC* parameterBoundaries,
	NTCALC EPSILON_1, NTCALC EPSILON_2) {
	int echoesCount = endIndex - startIndex + 1;
	int sliceSize = columnCount * rowCount;
		
	int length = sliceCount * sliceSize * echoesCount;
	NTCALC* y_values = new NTCALC[length];

	int resultLength = sliceCount * sliceSize * (parametersCount + 1);
	NTOUTPUT* result = new NTOUTPUT[resultLength];
	for(int index = 0; index < resultLength; ++index)
		result[index] = 0;

	NTCALC* tempCalculateError = new NTCALC[length];
	NTCALC* y_hat = new NTCALC[length];
	NTCALC* dydp = new NTCALC[length * parametersCount];
	
	for(int sliceIndex = 0; sliceIndex < sliceCount; ++sliceIndex)
		for(int index = 0; index < sliceSize; ++index) {
			int offsetInImage = sliceIndex * sliceSize + index;
			constants[0] = offsetInImage;
			CLevenbergMarquardtFletcher(
				startModelFunctionID,
				modelFunctionID,
				residualWeightingFunctionID,
				alphaWeightingFunctionID,
				parametersCount,
				offsetInImage, echoesCount, sliceSize, sliceCount,
				echotimes, data, threshold, constants, constantsLength, 
				parameterBoundaries, result,
				weights + echoesCount * sliceIndex, y_values, tempCalculateError, y_hat,
				dydp, EPSILON_1, EPSILON_2);
	}
	
	delete y_values;
	delete tempCalculateError;
	delete y_hat;
	delete dydp;

	return result;
}