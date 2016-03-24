/* 
 * File:   ProcessorContext.hpp
 * Author: christiantinauer
 *
 * Created on October 6, 2014, 4:24 PM
 */

#ifndef PROCESSORCONTEXT_HPP
#define	PROCESSORCONTEXT_HPP

class ProcessorContext {
 private:
	bool debug;
	bool useDoublePrecision;
	bool outputParameter2;
	bool outputParameter3;
	bool outputGoodnessOfFit;
	double threshold;
	int profileLength;
	int threadCount;
	double minGoF;
	double stepSizeTolerance;
	double errorReductionTolerance;
	
 public:
	ProcessorContext(bool debug, bool useDoublePrecision,
		bool outputParameter2, bool outputParameter3, 
		bool outputGoodnessOfFit, double threshold, int profileLength,
		int threadCount, double minGoF, double stepSizeTolerance,
		double errorReductionTolerance)
		: debug(debug), useDoublePrecision(useDoublePrecision),
			outputParameter2(outputParameter2),
			outputParameter3(outputParameter3),
			outputGoodnessOfFit(outputGoodnessOfFit),
			threshold(threshold), profileLength(profileLength),
			threadCount(threadCount), minGoF(minGoF),
			stepSizeTolerance(stepSizeTolerance), 
			errorReductionTolerance(errorReductionTolerance) {}
		
	inline bool getDebug() {
		return debug;
	}
	
	inline bool getUseDoublePrecision() {
		return useDoublePrecision;
	}
	
	inline bool getOutputParameter2() {
		return outputParameter2;
	}
	
	inline bool getOutputParameter3() {
		return outputParameter3;
	}
	
	inline bool getOutputGoodnessOfFit() {
		return outputGoodnessOfFit;
	}
	
	inline double getThreshold() {
		return threshold;
	}
	
	inline int getProfileLength() {
		return profileLength;
	}
	
	inline int getThreadCount() {
		return threadCount;
	}
	
	inline double getMinGoF() {
		return minGoF;
	}
	
	inline double getStepSizeTolerance() {
		return stepSizeTolerance;
	}
	
	inline double getErrorReductionTolerance() {
		return errorReductionTolerance;
	}
};

#endif	/* PROCESSORCONTEXT_HPP */