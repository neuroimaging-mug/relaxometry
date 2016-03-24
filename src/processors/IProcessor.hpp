#ifndef _Processor
#define _Processor

#include "../io/data/ProcessorData.hpp"
#include "../models/Model.hpp"
#include "ProcessorContext.hpp"

#include <string>
#include <vector>

using namespace std;

class IProcessor {
 public:
	virtual vector<ProcessorData*> Execute(Model* model,
		ProcessorContext* processorContext) = 0;
		
	virtual bool IsCUDAProcessor() = 0;
		
	static IProcessor* CreateProcessorFromArgument(const string& argument);
};

#endif