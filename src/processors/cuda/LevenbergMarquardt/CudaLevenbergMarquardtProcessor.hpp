/* 
 * File:   CudaLevenbergMarquardtProcessor.hpp
 * Author: christiantinauer
 *
 * Created on October 10, 2014, 10:55 AM
 */

#ifndef CUDALEVENBERGMARQUARDTPROCESSOR_HPP
#define	CUDALEVENBERGMARQUARDTPROCESSOR_HPP

#include "../../IProcessor.hpp"
#include "../../../io/data/ProcessorData.hpp"
#include "../../../models/Model.hpp"
#include "../../../processors/ProcessorContext.hpp"

class CudaLevenbergMarquardtProcessor : public IProcessor {
 public:
	CudaLevenbergMarquardtProcessor();

	~CudaLevenbergMarquardtProcessor();

	vector<ProcessorData*> Execute(Model* model,
		ProcessorContext* processorContext);
	
	bool IsCUDAProcessor();
};

#endif	/* CUDALEVENBERGMARQUARDTPROCESSOR_HPP */