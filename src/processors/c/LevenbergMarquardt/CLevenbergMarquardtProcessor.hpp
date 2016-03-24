/* 
 * File:   CLevenbergMarquardtWithExponentialModelProcessor.hpp
 * Author: christiantinauer
 *
 * Created on September 20, 2014, 4:01 PM
 */

#ifndef CLEVENBERGMARQUARDTWITHEXPONENTIALMODELPROCESSOR_H
#define	CLEVENBERGMARQUARDTWITHEXPONENTIALMODELPROCESSOR_H

#include "../../IProcessor.hpp"
#include "../../../io/data/ProcessorData.hpp"
#include "../../../models/Model.hpp"
#include "../../../processors/ProcessorContext.hpp"

class CLevenbergMarquardtProcessor : public IProcessor {
 public:
	CLevenbergMarquardtProcessor();
	
	~CLevenbergMarquardtProcessor();
	
	vector<ProcessorData*> Execute(Model* model,
		ProcessorContext* processorContext);
		
	bool IsCUDAProcessor();
};

#endif	/* CLEVENBERGMARQUARDTWITHEXPONENTIALMODELPROCESSOR_H */