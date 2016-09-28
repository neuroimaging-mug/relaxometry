#include "IProcessor.hpp"

#include "c/LinearRegression/CLinearRegressionProcessor.hpp"
#include "c/LevenbergMarquardt/CLevenbergMarquardtProcessor.hpp"
#include "c/LevenbergMarquardtFletcher/CLevenbergMarquardtFletcherProcessor.hpp"

#ifndef NO_CUDA
	#include "cuda/LinearRegression/CudaLinearRegressionProcessor.hpp"
	#include "cuda/LevenbergMarquardt/CudaLevenbergMarquardtProcessor.hpp"
	#include "cuda/LevenbergMarquardtFletcher/CudaLevenbergMarquardtFletcherProcessor.hpp"
#endif /* NO_CUDA */

#include <stdexcept>

IProcessor* IProcessor::CreateProcessorFromArgument(const string& argument) {
#ifndef NO_CUDA
	if(argument.compare("cudalr") == 0)
		return new CudaLinearRegressionProcessor();
	else if(argument.compare("cudalm") == 0)
		return new CudaLevenbergMarquardtProcessor();
	else if(argument.compare("cudalmf") == 0)
		return new CudaLevenbergMarquardtFletcherProcessor();
	else 
#endif /* NO_CUDA */
		
	if(argument.compare("clr") == 0)
		return new CLinearRegressionProcessor();
	else if(argument.compare("clm") == 0)
		return new CLevenbergMarquardtProcessor();
	else if(argument.compare("clmf") == 0)
		return new CLevenbergMarquardtFletcherProcessor();
	else
		throw runtime_error("Not supported argument for processor: " + argument);
}