/* 
 * File:   CudaLevenbergMarquardtProcessor.hpp
 * Author: christiantinauer
 *
 * Created on October 10, 2014, 10:55 AM
 */

#ifndef CUDALEVENBERGMARQUARDTFLETCHERPROCESSOR_HPP
#define	CUDALEVENBERGMARQUARDTFLETCHERPROCESSOR_HPP

#include "../../IProcessor.hpp"
#include "../../../io/data/ProcessorData.hpp"
#include "../../../models/Model.hpp"
#include "../../../processors/ProcessorContext.hpp"

class CudaLevenbergMarquardtFletcherProcessor : public IProcessor {
 public:
	CudaLevenbergMarquardtFletcherProcessor();

	~CudaLevenbergMarquardtFletcherProcessor();

	vector<ProcessorData*> Execute(Model* model,
		ProcessorContext* processorContext);
	
	bool IsCUDAProcessor();
};

#endif	/* CUDALEVENBERGMARQUARDTFLETCHERPROCESSOR_HPP */