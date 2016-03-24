#include "CLinearRegressionProcessor.hpp"

#include "../../../includes/StringExtensions.h"
#include "../../../models/Model.hpp"
#include "../../ProcessorService.hpp"

#include <stdexcept>
#include <string>

using namespace std;

#include "CLinearRegression.cpp"

CLinearRegressionProcessor::CLinearRegressionProcessor() {
}

CLinearRegressionProcessor::~CLinearRegressionProcessor() {
}

template<typename NTINPUT, typename NTOUTPUT, typename NTCALC>
vector<ProcessorData*> CLinearRegressionExecuteCore(Model* model, ProcessorContext* processorContext) {
	ProcessorData* processorData = model->getEchoesData();
	bool outputParameter2 = processorContext->getOutputParameter2();
	
	int count = processorData->getEndIndex() - processorData->getStartIndex() + 1;
	NTCALC echotimes[count];
	ProcessorService::CopyAndCast(model->getEchotimes(), echotimes, count);
	
	NTOUTPUT* output = CProcessLinearRegression<NTINPUT, NTOUTPUT, NTCALC>(
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
vector<ProcessorData*> CLinearRegressionExecuteCoreCaller(Model* model, ProcessorContext* processorContext) {
	ProcessorDataDataType outputDataType = model->getOutputDataType();
	if(processorContext->getUseDoublePrecision()) {
		if(outputDataType == INT16)
			return CLinearRegressionExecuteCore<NTINPUT, short, double>(model, processorContext);
		else if(outputDataType == FLOAT32)
			return CLinearRegressionExecuteCore<NTINPUT, float, double>(model, processorContext);
		else if(outputDataType == UINT16)
			return CLinearRegressionExecuteCore<NTINPUT, unsigned short, double>(model, processorContext);
		else
		throw runtime_error("Not supported output data type: " + to_string(outputDataType));
	} else {
		if(outputDataType == INT16)
			return CLinearRegressionExecuteCore<NTINPUT, short, float>(model, processorContext);
		else if(outputDataType == FLOAT32)
			return CLinearRegressionExecuteCore<NTINPUT, float, float>(model, processorContext);
		else if(outputDataType == UINT16)
			return CLinearRegressionExecuteCore<NTINPUT, unsigned short, float>(model, processorContext);
		else
			throw runtime_error("Not supported output data type: " + to_string(outputDataType));
	}
}

vector<ProcessorData*> CLinearRegressionProcessor::Execute(Model* model,
	ProcessorContext* processorContext) {
	ProcessorDataDataType dataType = model->getEchoesData()->getDataType();
	if(dataType == INT16)
		return CLinearRegressionExecuteCoreCaller<short>(model, processorContext);
	else if(dataType == FLOAT32)
		return CLinearRegressionExecuteCoreCaller<float>(model, processorContext);
	else if(dataType == UINT16)
		return CLinearRegressionExecuteCoreCaller<unsigned short>(model, processorContext);
	else
		throw runtime_error("Not supported data type: " + to_string(dataType));
}

bool CLinearRegressionProcessor::IsCUDAProcessor() {
	return false;
}