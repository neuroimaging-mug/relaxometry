/* 
 * File:   CudaLevenbergMarquardtProcessor.cpp
 * Author: christiantinauer
 * 
 * Created on October 10, 2014, 10:55 AM
 */
#include "CudaLevenbergMarquardtFletcherProcessor.hpp"

#include "../../../includes/StringExtensions.h"
#include "../../ProcessorService.hpp"
#include "../../../models/Model.hpp"
#include "CudaLevenbergMarquardtFletcher.hpp"

#include <stdexcept>
#include <string>

using namespace std;

CudaLevenbergMarquardtFletcherProcessor::CudaLevenbergMarquardtFletcherProcessor() {
}

CudaLevenbergMarquardtFletcherProcessor::~CudaLevenbergMarquardtFletcherProcessor() {
}

template<typename NTINPUT, typename NTOUTPUT, typename NTCALC>
vector<ProcessorData*> CudaLevenbergMarquardtFletcherExecuteCore(Model* model, ProcessorContext* processorContext) {
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
	int constantsLength = 7;
	if(model->getNeedsFlipAnglesMap())
		constantsLength += dataLength;
	if(model->getNeedsT1Map())
		constantsLength += dataLength;
	NTCALC constants[constantsLength];
	constants[0] = startValues[0];
	constants[1] = startValues[1];
	constants[2] = parametersCount == 3 ? startValues[2] : 0.f;
	constants[3] = processorData->getEchospacing();
	constants[4] = processorContext->getProfileLength();
	constants[5] = dataLength;
	constants[6] = 128.f;
	if(model->getNeedsFlipAnglesMap()) {
		NTCALC* flipAnglesData = constants + 7;
		if(model->getFlipAnglesData()->getDataType() == FLOAT64)
			ProcessorService::CopyAndCast((double*)model->getFlipAnglesData()->getData(), flipAnglesData, dataLength);
		else
			ProcessorService::CopyAndCast((float*)model->getFlipAnglesData()->getData(), flipAnglesData, dataLength);
	}
	if(model->getNeedsT1Map()) {
		NTCALC* t1Data = constants + 7 + dataLength;
		if(model->getT1Data()->getDataType() == FLOAT64)
			ProcessorService::CopyAndCast((double*)model->getT1Data()->getData(), t1Data, dataLength);
		else
			ProcessorService::CopyAndCast((float*)model->getT1Data()->getData(), t1Data, dataLength);
	}
	
	//Process
	NTOUTPUT* output = CudaProcessLevenbergMarquardtFletcher<NTINPUT, NTOUTPUT, NTCALC>(
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
		model->getNeedsFlipAnglesMap(),
		model->getNeedsT1Map(),
		processorContext->getThreadCount(),
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
vector<ProcessorData*> CudaLevenbergMarquardtFletcherExecuteCoreCaller(Model* model, ProcessorContext* processorContext) {
	ProcessorDataDataType outputDataType = model->getOutputDataType();
	if(outputDataType == INT16)
		return CudaLevenbergMarquardtFletcherExecuteCore<NTINPUT, short, float>(model, processorContext);
	else if(outputDataType == FLOAT32)
		return CudaLevenbergMarquardtFletcherExecuteCore<NTINPUT, float, float>(model, processorContext);
	else if(outputDataType == UINT16)
		return CudaLevenbergMarquardtFletcherExecuteCore<NTINPUT, unsigned short, float>(model, processorContext);
	else
		throw runtime_error("Not supported output data type: " + to_string(outputDataType));
}

vector<ProcessorData*> CudaLevenbergMarquardtFletcherProcessor::Execute(Model* model,
	ProcessorContext* processorContext) {
	ProcessorDataDataType dataType = model->getEchoesData()->getDataType();
	if(dataType == INT16)
		return CudaLevenbergMarquardtFletcherExecuteCoreCaller<short>(model, processorContext);
	else if(dataType == FLOAT32)
		return CudaLevenbergMarquardtFletcherExecuteCoreCaller<float>(model, processorContext);
	else if(dataType == UINT16)
		return CudaLevenbergMarquardtFletcherExecuteCoreCaller<unsigned short>(model, processorContext);
	else
		throw runtime_error("Not supported data type: " + to_string(dataType));
}

bool CudaLevenbergMarquardtFletcherProcessor::IsCUDAProcessor() {
	return true;
}