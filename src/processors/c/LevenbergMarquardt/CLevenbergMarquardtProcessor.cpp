/* 
 * File:   CLevenbergMarquardtProcessor.cpp
 * Author: christiantinauer
 * 
 * Created on September 20, 2014, 4:01 PM
 */

#include "CLevenbergMarquardtProcessor.hpp"

#include "../../../includes/StringExtensions.h"
#include "../../ProcessorService.hpp"
#include "../../../models/Model.hpp"

#include <stdexcept>
#include <string>

using namespace std;

#include "CLevenbergMarquardt.cpp"

CLevenbergMarquardtProcessor::CLevenbergMarquardtProcessor() {
}

CLevenbergMarquardtProcessor::~CLevenbergMarquardtProcessor() {
}

template<typename NTINPUT, typename NTOUTPUT, typename NTCALC>
vector<ProcessorData*> CLevenbergMarquardtExecuteCore(Model* model, ProcessorContext* processorContext) {
	ProcessorData* processorData = model->getEchoesData();
	bool outputParameter2 = processorContext->getOutputParameter2();
	bool outputParameter3 = processorContext->getOutputParameter3();
	bool outputGoodnessOfFit = processorContext->getOutputGoodnessOfFit();
	
	//startvalues
	int parametersCount = model->getParametersCount();
	double* startValues = model->getParameterStartValues();
	
	//echotimes
	int echoCount = processorData->getEndIndex() - processorData->getStartIndex() + 1;
	NTCALC echotimes[echoCount];
	ProcessorService::CopyAndCast(model->getEchotimes(), echotimes, echoCount);
	
	//weights
	int weightCount = echoCount * processorData->getSliceCount();
	NTCALC weights[weightCount];
	ProcessorService::CopyAndCast(model->getWeights(), weights, weightCount);
	
	//boundaries
	int boundariesCount = parametersCount * 2;
	NTCALC boundaries[boundariesCount];
	ProcessorService::CopyAndCast(model->getParameterBoundaries(), boundaries, boundariesCount);
	
	//constants
	int dataLength = 
		processorData->getColumnCount() * 
		processorData->getRowCount() * 
		processorData->getSliceCount();
	int constantsLength = 8;
	if(model->getNeedsFlipAnglesMap())
		constantsLength += dataLength;
	if(model->getNeedsT1Map())
		constantsLength += dataLength;
	NTCALC constants[constantsLength];
	constants[0] = -1; //index
	constants[1] = startValues[0];
	constants[2] = startValues[1];
	constants[3] = parametersCount > 2 ? startValues[2] : 0;
	constants[4] = processorData->getEchospacing();
	constants[5] = processorContext->getProfileLength();
	constants[6] = dataLength;
	constants[7] = 128.f;
	if(model->getNeedsFlipAnglesMap()) {
		NTCALC* flipAnglesData = constants + 8;
		if(model->getFlipAnglesData()->getDataType() == FLOAT64)
			ProcessorService::CopyAndCast((double*)model->getFlipAnglesData()->getData(), flipAnglesData, dataLength);
		else
			ProcessorService::CopyAndCast((float*)model->getFlipAnglesData()->getData(), flipAnglesData, dataLength);
	}
	if(model->getNeedsT1Map()) {
		NTCALC* t1Data = constants + 8 + dataLength;
		if(model->getT1Data()->getDataType() == FLOAT64)
			ProcessorService::CopyAndCast((double*)model->getT1Data()->getData(), t1Data, dataLength);
		else
			ProcessorService::CopyAndCast((float*)model->getT1Data()->getData(), t1Data, dataLength);
	}
	
	//Process
	NTOUTPUT* output = CProcessLevenbergMarquardt<NTINPUT, NTOUTPUT, NTCALC>(
		model->getStartModelFunctionID(),
		model->getModelFunctionID(),
		model->getResidualWeightingFunctionID(),
		model->getAlphaWeightingFunctionID(),
		parametersCount,
		processorData->getStartIndex(),
		processorData->getEndIndex(),
		processorData->getColumnCount(),
		processorData->getRowCount(),
		processorData->getSliceCount(),
		echotimes,
		weights,
		(NTINPUT*)processorData->getData(),
		(NTCALC)processorContext->getThreshold(),
		constants,
		constantsLength,
		boundaries,
		(NTCALC)processorContext->getStepSizeTolerance(),
		(NTCALC)processorContext->getErrorReductionTolerance());
	
	NTOUTPUT* parameter2 = NULL;
	if(outputParameter2)
		parameter2 = output + 
			processorData->getSliceCount() *
			processorData->getColumnCount() *
			processorData->getRowCount();
	
	NTOUTPUT* parameter3 = NULL;
	if(outputParameter3)
		parameter3 = parameter2 + 
			processorData->getSliceCount() *
			processorData->getColumnCount() *
			processorData->getRowCount();
	
	NTOUTPUT* goodnessOfFit = NULL;
	if(outputGoodnessOfFit)
		goodnessOfFit = (outputParameter3 ? parameter3 : parameter2) + 
			processorData->getSliceCount() *
			processorData->getColumnCount() *
			processorData->getRowCount();
			
	vector<ProcessorData*> outputs;
	outputs.push_back(
		new ProcessorData(
			processorData->getStartIndex(),
			processorData->getEndIndex(),
			processorData->getColumnCount(),
			processorData->getRowCount(),
			processorData->getSliceCount(),
			processorData->getEchospacing(),
			output,
			model->getOutputDataType()));
	if(parameter2 != NULL)
		outputs.push_back(
			new ProcessorData(
				processorData->getStartIndex(),
				processorData->getEndIndex(),
				processorData->getColumnCount(),
				processorData->getRowCount(),
				processorData->getSliceCount(),
				processorData->getEchospacing(),
				parameter2,
				model->getOutputDataType()));
		if(parameter3 != NULL)
		outputs.push_back(
			new ProcessorData(
				processorData->getStartIndex(),
				processorData->getEndIndex(),
				processorData->getColumnCount(),
				processorData->getRowCount(),
				processorData->getSliceCount(),
				processorData->getEchospacing(),
				parameter3,
				model->getOutputDataType()));
	if(goodnessOfFit != NULL)
		outputs.push_back(
			new ProcessorData(
				processorData->getStartIndex(),
				processorData->getEndIndex(),
				processorData->getColumnCount(),
				processorData->getRowCount(),
				processorData->getSliceCount(),
				processorData->getEchospacing(),
				goodnessOfFit,
				model->getOutputDataType()));
	return outputs;
}

template<typename NTINPUT>
vector<ProcessorData*> CLevenbergMarquardtExecuteCoreCaller(Model* model, ProcessorContext* processorContext) {
	ProcessorDataDataType outputDataType = model->getOutputDataType();
	if(processorContext->getUseDoublePrecision()) {
		if(outputDataType == INT16)
			return CLevenbergMarquardtExecuteCore<NTINPUT, short, double>(model, processorContext);
		else if(outputDataType == FLOAT32)
			return CLevenbergMarquardtExecuteCore<NTINPUT, float, double>(model, processorContext);
		else if(outputDataType == UINT16)
			return CLevenbergMarquardtExecuteCore<NTINPUT, unsigned short, double>(model, processorContext);
		else
			throw runtime_error("Not supported output data type: " + to_string(outputDataType));
	} else {
		if(outputDataType == INT16)
			return CLevenbergMarquardtExecuteCore<NTINPUT, short, float>(model, processorContext);
		else if(outputDataType == FLOAT32)
			return CLevenbergMarquardtExecuteCore<NTINPUT, float, float>(model, processorContext);
		else if(outputDataType == UINT16)
			return CLevenbergMarquardtExecuteCore<NTINPUT, unsigned short, float>(model, processorContext);
		else
			throw runtime_error("Not supported output data type: " + to_string(outputDataType));
	}
}

vector<ProcessorData*> CLevenbergMarquardtProcessor::Execute(Model* model,
	ProcessorContext* processorContext) {
	ProcessorDataDataType dataType = model->getEchoesData()->getDataType();
	if(dataType == INT16)
		return CLevenbergMarquardtExecuteCoreCaller<short>(model, processorContext);
	else if(dataType == FLOAT32)
		return CLevenbergMarquardtExecuteCoreCaller<float>(model, processorContext);
	else if(dataType == UINT16)
		return CLevenbergMarquardtExecuteCoreCaller<unsigned short>(model, processorContext);
	else
		throw runtime_error("Not supported data type: " + to_string(dataType));
}

bool CLevenbergMarquardtProcessor::IsCUDAProcessor() {
	return false;
}