#include "CudaLinearRegressionProcessor.hpp"

#include "../../../includes/StringExtensions.h"
#include "../../ProcessorService.hpp"
#include <stdexcept>
#include <string>

#include "CudaLinearRegression.hpp"

CudaLinearRegressionProcessor::CudaLinearRegressionProcessor() {
}

CudaLinearRegressionProcessor::~CudaLinearRegressionProcessor() {
}

template<typename NTINPUT, typename NTOUTPUT, typename NTCALC>
vector<ProcessorData*> CudaLinearRegressionExecuteCore(Model* model, ProcessorContext* processorContext) {
	ProcessorData* processorData = model->getEchoesData();
	bool outputParameter2 = processorContext->getOutputParameter2();
	
	int count = processorData->getEndIndex() - processorData->getStartIndex() + 1;
	NTCALC echotimes[count];
	ProcessorService::CopyAndCast(model->getEchotimes(), echotimes, count);
	
	NTOUTPUT* output = CudaProcessLinearRegression<NTINPUT, NTOUTPUT, NTCALC>(
		model->getModelFunctionID(),
		processorData->getStartIndex(),
		processorData->getEndIndex(),
		processorData->getColumnCount(),
		processorData->getRowCount(),
		processorData->getSliceCount(),
		echotimes,
		(NTINPUT*)processorData->getData(),
		(NTCALC)processorContext->getThreshold());
	
	NTOUTPUT* parameter2 = NULL;
	if(outputParameter2)
		parameter2 = output + 
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
	return outputs;
}

template<typename NTINPUT>
vector<ProcessorData*> CudaLinearRegressionExecuteCoreCaller(Model* model, ProcessorContext* processorContext) {
	ProcessorDataDataType outputDataType = model->getOutputDataType();
	if(outputDataType == INT16)
		return CudaLinearRegressionExecuteCore<NTINPUT, short, float>(model, processorContext);
	else if(outputDataType == FLOAT32)
		return CudaLinearRegressionExecuteCore<NTINPUT, float, float>(model, processorContext);
	else if(outputDataType == UINT16)
		return CudaLinearRegressionExecuteCore<NTINPUT, unsigned short, float>(model, processorContext);
	else
		throw runtime_error("Not supported output data type: " + to_string(outputDataType));
}

vector<ProcessorData*> CudaLinearRegressionProcessor::Execute(Model* model,
	ProcessorContext* processorContext) {
	ProcessorDataDataType dataType = model->getEchoesData()->getDataType();
	if(dataType == INT16)
		return CudaLinearRegressionExecuteCoreCaller<short>(model, processorContext);
	else if(dataType == FLOAT32)
		return CudaLinearRegressionExecuteCoreCaller<float>(model, processorContext);
	else if(dataType == UINT16)
		return CudaLinearRegressionExecuteCoreCaller<unsigned short>(model, processorContext);
	else
		throw runtime_error("Not supported data type: " + to_string(dataType));
}

bool CudaLinearRegressionProcessor::IsCUDAProcessor() {
	return true;
}