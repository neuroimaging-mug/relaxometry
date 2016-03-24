/* 
 * File:   CLevenbergMarquardtWithLukzenSavlovModelProcessor.hpp
 * Author: christiantinauer
 *
 * Created on October 10, 2014, 10:55 AM
 */

#ifndef CLEVENBERGMARQUARDTFLETCHERPROCESSOR_HPP
#define	CLEVENBERGMARQUARDTFLETCHERPROCESSOR_HPP

#include "../../IProcessor.hpp"
#include "../../../io/data/ProcessorData.hpp"
#include "../../../models/Model.hpp"
#include "../../../processors/ProcessorContext.hpp"

class CLevenbergMarquardtFletcherProcessor : public IProcessor {
 public:
	CLevenbergMarquardtFletcherProcessor();
	
	~CLevenbergMarquardtFletcherProcessor();
	
	vector<ProcessorData*> Execute(Model* model,
		ProcessorContext* processorContext);
		
	bool IsCUDAProcessor();
};

#endif	/* CLEVENBERGMARQUARDTFLETCHERPROCESSOR_HPP */