/* 
 * File:   ProcessorService.hpp
 * Author: christiantinauer
 *
 * Created on September 12, 2014, 6:57 PM
 */

#ifndef PROCESSORSERVICE_HPP
#define	PROCESSORSERVICE_HPP

#include <cmath>

using namespace std;

class ProcessorService {
 private:
	ProcessorService();

 public:
	template<typename NUMERICTYPE>
	inline static NUMERICTYPE GetMinValueOfData(
		int sliceCount, int columnCount, int rowCount, NUMERICTYPE* data) {
		int sliceSize = rowCount * columnCount;
		NUMERICTYPE min = 0;

		for (int slice = 0; slice < sliceCount; ++slice)
			for (int x = 0; x < columnCount; ++x)
				for (int y = 0; y < rowCount; ++y) {
					int offsetInSlice = y * columnCount + x;
					NUMERICTYPE candidate = data[slice * sliceSize + offsetInSlice];
					if (candidate < min)
						min = candidate;
				}

		return min;
	}

	template<typename NUMERICTYPE>
	inline static NUMERICTYPE CalculateAdjustedRsquareValue(NUMERICTYPE* y,
		NUMERICTYPE* y_hat, int length, int parametersLength) {
		NUMERICTYPE ySum = 0;
		NUMERICTYPE SSE = 0;
		for(int index = 0; index < length; ++index) {
			ySum += y[index];
			NUMERICTYPE error = y[index] - y_hat[index];
			SSE += error * error;
		}
		NUMERICTYPE yMean = ySum / (NUMERICTYPE)length;
		NUMERICTYPE SST = 0;
		for(int index = 0; index < length; ++index) {
			NUMERICTYPE diff = y[index] - yMean;
			SST += diff * diff;
		}
		NUMERICTYPE adjustedRsquare = 1 - 
			(length - 1) / (length - parametersLength) *
			SSE / SST;
//		if(adjustedRsquare < 0)
//			adjustedRsquare = 0;
		return adjustedRsquare;
	}
	
	template<typename NUMERICTYPE>
	inline static NUMERICTYPE MaxAbs(NUMERICTYPE* values, int length) {
		NUMERICTYPE max = 0;
		for (int i = 0; i < length; ++i) {
			float value = abs(values[i]);
			if (value > max)
				max = value;
		}
		return max;
	}
	
	template<typename NTINPUT, typename NTOUTPUT>
	inline static void CopyAndCast(NTINPUT* source, NTOUTPUT* destination, int length) {
		for(int index = 0; index < length; ++index)
			destination[index] = (NTOUTPUT)source[index];
	}
};

#endif	/* PROCESSORSERVICE_HPP */