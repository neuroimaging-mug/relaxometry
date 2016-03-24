/* 
 * File:   DataTransfer.cuh
 * Author: christiantinauer
 *
 * Created on September 18, 2015, 4:25 PM
 */

#ifndef DATATRANSFER_CUH
#define	DATATRANSFER_CUH

template<typename NTDATA>
inline int MarkValidData(NTDATA* data, bool* validData, int length, float threshold) {
	int validDataCount = 0;
	for(int index = 0; index < length; ++index) {
		bool isValid = data[index] > threshold;
		validData[index] = isValid;
		if(isValid)
			++validDataCount;
	}
	return validDataCount;
}

template<typename NTDATA, typename NTCALC>
void RestructureDataForward(
	NTDATA* echoesData, NTCALC* flipAnglesData, NTCALC* t1Data,
	bool* validData, int length, int sliceSize, int echoesCount, NTCALC* restructeredData) {
	int validDataIndex = 0;
	for(int dataIndex = 0; dataIndex < length; ++dataIndex) 
		if(validData[dataIndex]) {
			//time series
			for(int echoIndex = 0; echoIndex < echoesCount; ++echoIndex)
				restructeredData[validDataIndex++] = 
					echoesData[dataIndex + echoIndex * length];
			//slice index
			restructeredData[validDataIndex++] = (NTCALC)((int)(dataIndex / sliceSize));
			//flipangle
			if(flipAnglesData != NULL)
				restructeredData[validDataIndex++] = flipAnglesData[dataIndex];
			//t1
			if(t1Data != NULL)
				restructeredData[validDataIndex++] = t1Data[dataIndex];
		}
}

template<typename NTDATA, typename NTCALC>
void RestructureDataBackward(
NTDATA* data, bool* validData, int dataLength, NTCALC* restructeredData,
	int parametersCount, bool calculateGoodnessOfFit) {
	int restructeredDataIndex = 0;
	for(int index = 0; index < dataLength; ++index)
		if(validData[index]) {
			for(int innerIndex = 0; innerIndex < parametersCount; ++innerIndex)
				data[index + innerIndex * dataLength] = restructeredData[restructeredDataIndex++];
			if(calculateGoodnessOfFit)
				data[index + parametersCount * dataLength] = restructeredData[restructeredDataIndex++];
		} else {
			for(int innerIndex = 0; innerIndex < parametersCount; ++innerIndex)
				data[index + innerIndex * dataLength] = 0;
			if(calculateGoodnessOfFit)
				data[index + parametersCount * dataLength] = 0;
		}
}

#endif	/* DATATRANSFER_CUH */