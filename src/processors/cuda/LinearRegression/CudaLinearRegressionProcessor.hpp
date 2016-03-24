#ifndef _CudaLinearRegressionProcessor
#define _CudaLinearRegressionProcessor

#include "../../IProcessor.hpp"
#include "../../../io/data/ProcessorData.hpp"
#include "../../../models/Model.hpp"
#include "../../../processors/ProcessorContext.hpp"

class CudaLinearRegressionProcessor : public IProcessor {
 public:
	CudaLinearRegressionProcessor();

	~CudaLinearRegressionProcessor();

	vector<ProcessorData*> Execute(Model* model,
		ProcessorContext* processorContext);
	
	bool IsCUDAProcessor();
};

#endif