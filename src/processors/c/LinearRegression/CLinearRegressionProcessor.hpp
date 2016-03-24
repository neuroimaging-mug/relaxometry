#ifndef _CLinearRegressionProcessor
#define _CLinearRegressionProcessor

#include "../../IProcessor.hpp"
#include "../../../io/data/ProcessorData.hpp"
#include "../../../models/Model.hpp"
#include "../../../processors/ProcessorContext.hpp"

class CLinearRegressionProcessor : public IProcessor {
 public:
	CLinearRegressionProcessor();
	
	~CLinearRegressionProcessor();
	
	vector<ProcessorData*> Execute(Model* model,
		ProcessorContext* processorContext);
		
	bool IsCUDAProcessor();
};

#endif