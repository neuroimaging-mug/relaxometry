/* 
 * File:   ProcessorData.hpp
 * Author: christiantinauer
 *
 * Created on August 27, 2014, 10:57 AM
 */

#ifndef PROCESSORDATA_HPP
#define	PROCESSORDATA_HPP

//#define NIFTI_TYPE_UINT8           2 /! unsigned char. /
//#define NIFTI_TYPE_INT16           4 /! signed short. /
//#define NIFTI_TYPE_INT32           8 /! signed int. /
//#define NIFTI_TYPE_FLOAT32        16 /! 32 bit float. /
//#define NIFTI_TYPE_COMPLEX64      32 /! 64 bit complex = 2 32 bit floats. /
//#define NIFTI_TYPE_FLOAT64        64 /! 64 bit float = double. /
//#define NIFTI_TYPE_RGB24         128 /! 3 8 bit bytes. /
//#define NIFTI_TYPE_INT8          256 /! signed char. /
//#define NIFTI_TYPE_UINT16        512 /! unsigned short. /
//#define NIFTI_TYPE_UINT32        768 /! unsigned int. /
//#define NIFTI_TYPE_INT64        1024 /! signed long long. /
//#define NIFTI_TYPE_UINT64       1280 /! unsigned long long. /
//#define NIFTI_TYPE_FLOAT128     1536 /! 128 bit float = long double. /
//#define NIFTI_TYPE_COMPLEX128   1792 /! 128 bit complex = 2 64 bit floats. /
//#define NIFTI_TYPE_COMPLEX256   2048 /! 256 bit complex = 2 128 bit floats /

enum ProcessorDataDataType {
	INT16 = 4,
	FLOAT32 = 16,
	FLOAT64 = 64,
	UINT16 = 512
};

class ProcessorData {
 private:
	int startIndex;
	int endIndex;
	int rowCount;
	int columnCount;
	int sliceCount;
	float echospacing;
	void* data;
	ProcessorDataDataType dataType;

 public:
	ProcessorData(
		int startIndex, int endIndex, int columnCount, int rowCount, 
		int sliceCount, float echospacing,
		void* data, ProcessorDataDataType dataType)
		: startIndex(startIndex), endIndex(endIndex), columnCount(columnCount),
			rowCount(rowCount), sliceCount(sliceCount), echospacing(echospacing),
			data(data), dataType(dataType) {}

	~ProcessorData() {}

	inline int getStartIndex() {
		return startIndex;
	}

	inline int getEndIndex() {
		return endIndex;
	}

	inline int getColumnCount() {
		return columnCount;
	}

	inline int getRowCount() {
		return rowCount;
	}
	
	inline int getSliceCount() {
		return sliceCount;
	}

	inline float getEchospacing() {
		return echospacing;
	}
	
	inline void* getData() {
		return data;
	}
	
	inline ProcessorDataDataType getDataType() {
		return dataType;
	}
};

#endif	/* PROCESSORDATA_HPP */