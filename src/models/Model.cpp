#include "Model.hpp"

//#include "c/ModelService.hpp"

#include <stdexcept>

short ArgumentToModelFunctionID(const string& argument) {
	if(argument.compare("exp2pr1") == 0)
		return 0;
	else if(argument.compare("exp2pt1") == 0)
		return 1;
	else if(argument.compare("exp3pr1") == 0)
		return 2;
	else if(argument.compare("exp3pt1") == 0)
		return 3;
	else if(argument.compare("expr2") == 0)
		return 4;
	else if(argument.compare("expt2") == 0)
		return 5;
	else if(argument.compare("ls") == 0)
		return 6;
	else if(argument.compare("lrr2") == 0)
		return 7;
	else if(argument.compare("lrt2") == 0)
		return 8;
	else if(argument.compare("ssv") == 0)
		return 9;
	else
		return -1;
}

short ModelFunctionIDToStartModelFunctionID(const short id) {
	switch(id) {
		case 0: return 9;
		case 1: return 9;
		case 2: return 9;
		case 3: return 9;
		case 4: return 7;
		case 5: return 8;
		case 6: return 8;
		default: return -1;
	}
}

//ModelFunction ModelFunctionIDToModelFunction(const short id) {
//	switch(id) {
//		case 0: return &ModelService::Exponential2ParametersR1<float>;
//		case 1: return &ModelService::Exponential2ParametersT1<float>;
//		case 2: return &ModelService::Exponential3ParametersR1<float>;
//		case 3: return &ModelService::Exponential3ParametersT1<float>;
//		case 4: return &ModelService::ExponentialR2<float>;
//		case 5: return &ModelService::ExponentialT2<float>;
//		case 6: return &ModelService::LukzenSavelov<float>;
//		case 7: return &ModelService::LinearRegressionR2<float>;
//		case 8: return &ModelService::LinearRegressionT2<float>;
//		case 9: return &ModelService::SimpleStartValues<float>;
//		default: return NULL;
//	}
//}

int ModelFunctionIDToParametersCount(const short id) {
	if(id == 2 || id == 3)
		return 3;
	else
		return 2;
}

float ModelFunctionIDToTimeCorrectionFactor(const short id) {
	if(id == 0 || id == 2 || id == 4 || id == 7)
		return 0.001f;
	else
		return 1.f;
}

short ModelFunctionIDToResidualWeightingFunctionID(const short id) {
	switch(id) {
		case 4: return 1;
		case 5: return 1;
		case 6: return 1;
		default: return -1;
	}
}

short ModelFunctionIDToAlphaWeightingFunctionID(const short id) {
//	switch(id) {
//		case 4: return 2;
//		case 5: return 2;
//		case 6: return 2;
//		default: return 0;
//	}
	return -1;
}

//WeightingFunction WeightingFunctionIDToWeightingFunction(const short id) {
//	switch(id) {
//		case 0: return &ModelService::LinearWeighting<double>;
//		case 1: return &ModelService::InverseMinimumWeighting<double>;
//		case 2: return &ModelService::InverseQuadraticWeighting<double>;
//		default: return NULL;
//	}
//}

Model* Model::CreateModelFromArgument(const string& argument, bool forCUDAProcessor,
	bool useWeights) {
	short modelFunctionID = ArgumentToModelFunctionID(argument);
	short startModelFunctionID = ModelFunctionIDToStartModelFunctionID(modelFunctionID);
	short residualWeightingFunctionID = useWeights ? ModelFunctionIDToResidualWeightingFunctionID(modelFunctionID) : -1;
	short alphaWeightingFunctionID = useWeights ? ModelFunctionIDToAlphaWeightingFunctionID(modelFunctionID) : -1;
//	ModelFunction startModelFunction = forCUDAProcessor ? NULL : ModelFunctionIDToModelFunction(startModelFunctionID);
//	ModelFunction modelFunction = forCUDAProcessor ? NULL : ModelFunctionIDToModelFunction(modelFunctionID);
//	WeightingFunction residualWeightingFunction = forCUDAProcessor ? NULL : WeightingFunctionIDToWeightingFunction(residualWeightingFunctionID);
//	WeightingFunction alphaWeightingFunction = forCUDAProcessor ? NULL : WeightingFunctionIDToWeightingFunction(alphaWeightingFunctionID);
	
	int parametersCount = ModelFunctionIDToParametersCount(modelFunctionID);
	float timeCorrectionFactor = ModelFunctionIDToTimeCorrectionFactor(modelFunctionID);
	if(modelFunctionID == 0 || modelFunctionID == 1)
		return new Model(
			startModelFunctionID, modelFunctionID,
			residualWeightingFunctionID,
			alphaWeightingFunctionID,
			parametersCount, timeCorrectionFactor, true, false, FLOAT32);
	else if(modelFunctionID == 6)
		return new Model(
			startModelFunctionID, modelFunctionID,
			residualWeightingFunctionID,
			alphaWeightingFunctionID,
			parametersCount, timeCorrectionFactor, true, true, FLOAT32);
	else
		return new Model(
			startModelFunctionID, modelFunctionID,
			residualWeightingFunctionID,
			alphaWeightingFunctionID,
			parametersCount, timeCorrectionFactor, false, false, FLOAT32);
}