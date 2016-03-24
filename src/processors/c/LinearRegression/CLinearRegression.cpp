#include "../../../models/c/Model.hpp"

#include <cmath>

using namespace std;

template<typename NTINPUT, typename NTOUTPUT, typename NTCALC>
NTOUTPUT* CProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	NTCALC* echotimes, NTINPUT* data, NTCALC threshold) {
	int parametersLength = 2;
	NTCALC parameters[parametersLength];
	int sliceSize = columnCount * rowCount;
	int dataPerSlice = sliceSize * parametersLength;
	NTOUTPUT* output = new NTOUTPUT[sliceCount * dataPerSlice];
	int echoCount = endIndex - startIndex + 1;
	
	for(int slice = 0; slice < sliceCount; ++slice)
		for(int y = 0; y < rowCount; ++y)
			for(int x = 0; x < columnCount; ++x) {
				int dataIndex = slice * sliceSize + y * columnCount + x;
				if(data[dataIndex] <= threshold) {
					parameters[0] = 0;
					parameters[1] = 0;
				} else {
					NTCALC y_values[echoCount];
					for(int echoIndex = 0; echoIndex < echoCount; ++echoIndex)
						y_values[echoIndex] = data[dataIndex + echoIndex * sliceCount * sliceSize];
					
					cModelFunction(modelFunctionID, echotimes, y_values, echoCount, 
						parameters, parametersLength, (NTCALC*)NULL, 0);
				}
				
				//p1 (T1/T2)
				output[dataIndex] = (NTOUTPUT)parameters[0];
	
				//p2 (M0)
				output[dataIndex + sliceCount * sliceSize] = (NTOUTPUT)parameters[1];
			}
	
	return output;
}