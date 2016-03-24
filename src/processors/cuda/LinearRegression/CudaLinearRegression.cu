#include "../../../cuda/CUDADevicesService.hpp"
#include "../../../datatransfer/DataTransfer.cuh"
#include "../../../models/cuda/Statics.cuh"
#include "../../../models/cuda/Model.cuh"
#include "../ErrorHandling.cuh"
#include "../MemoryHandling.cuh"

#include <cuda_runtime.h>

template<typename NTOUTPUT, typename NTCALC>
__global__ void CudaLinearRegression(
	short modelFunctionID,
	NTCALC* x_values, NTOUTPUT* output, int length, int echoesCount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= length)
		return;
	
	int parametersLength = 2;
	NTCALC parameters[2];

	cudaModelFunction(
		modelFunctionID,
		x_values, (NTCALC*)NULL, echoesCount, 
		parameters, parametersLength, (NTCALC*)NULL, 0);
	
	__syncthreads();
	
	//p1 (T1/T2)
	output[index * 2] = (NTOUTPUT)parameters[0];
	
	//p2 (M0)
	output[index * 2 + 1] = (NTOUTPUT)parameters[1];
}

template<typename NTINPUT, typename NTOUTPUT, typename NTCALC>
NTOUTPUT* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	NTCALC* echotimes, NTINPUT* data, NTCALC threshold) {
	int sliceSize = columnCount * rowCount;
	int dataLength = sliceSize * sliceCount;
	int echoesCount = endIndex - startIndex + 1;
	int parametersCount = 2;
	
	//mark valid data
	bool validData[dataLength];
	int validDataLength = MarkValidData(data, validData, dataLength, threshold);
	
	//restructure data
	int restructeredColumnsCount = echoesCount + 1; //sliceIndex
	NTCALC* restructeredData = new NTCALC[validDataLength * restructeredColumnsCount];
	RestructureDataForward(data, (NTCALC*)NULL, (NTCALC*)NULL,
		validData, dataLength, sliceSize, echoesCount, restructeredData);
	
	int maxTexture2DHeight = CUDADevicesService::getMaximumTexture2DHeight(); 
	int kernelCalls = validDataLength / maxTexture2DHeight;
	if(validDataLength % maxTexture2DHeight != 0)
		++kernelCalls;
	int restructeredRowsCount = min(maxTexture2DHeight, validDataLength);

	//processConstants
	float constBuilder[263] = {};
	for(int index = 0; index < echoesCount; ++index)
		constBuilder[index + 13] = echotimes[index];
	HANDLE_CUDA_ERROR(cudaMemcpyToSymbol(processConstants, constBuilder,
		263 * sizeof(float), 0, cudaMemcpyHostToDevice));
	
	NTCALC* gpu_x_values = NULL; //AllocAndCopyToDevice(echotimes, echoesCount);
	
	NTOUTPUT* gpu_output = 
		AllocOnDevice<NTOUTPUT>(validDataLength * parametersCount);
	
	//CudaArray + Texture for restructered data
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<NTCALC>();
	
	cudaExtent restructeredDataExtent = 
		make_cudaExtent(restructeredColumnsCount, restructeredRowsCount, 0);
	
	cudaArray* dataArray = NULL;
	HANDLE_CUDA_ERROR(cudaMalloc3DArray(
		&dataArray, &channelFormatDesc, restructeredDataExtent, cudaArrayDefault));
	
	int threadsPerBlock = 128;
	for(int kernelIndex = 0; kernelIndex < kernelCalls; ++kernelIndex) {
		int remainingValidDataLength = 
			validDataLength - kernelIndex * restructeredRowsCount;
		int callRowsCount = min(restructeredRowsCount, remainingValidDataLength);
		
		HANDLE_CUDA_ERROR(cudaMemcpyToArray(dataArray, 0, 0, 
			restructeredData + kernelIndex * restructeredColumnsCount * restructeredRowsCount, 
			restructeredColumnsCount * sizeof(NTCALC) * callRowsCount,
			cudaMemcpyHostToDevice));
		HANDLE_CUDA_ERROR(
			cudaBindTextureToArray(floatTexture, dataArray, channelFormatDesc));
		
		int blockCount = (callRowsCount + threadsPerBlock - 1) / threadsPerBlock;
		CudaLinearRegression<<<blockCount, threadsPerBlock>>>(
			modelFunctionID,
			gpu_x_values, gpu_output + kernelIndex * restructeredRowsCount * parametersCount, 
			callRowsCount, echoesCount);
		
		HANDLE_CUDA_ERROR(cudaGetLastError());
		HANDLE_CUDA_ERROR(cudaUnbindTexture(floatTexture));
	}
	
  NTOUTPUT* output = new NTOUTPUT[dataLength * parametersCount];
	NTOUTPUT* restructeredOutput = CopyFromDeviceAndFree(gpu_output, validDataLength * parametersCount);
	HANDLE_CUDA_ERROR(cudaGetLastError());
	RestructureDataBackward(output, validData, dataLength, restructeredOutput,
		parametersCount, false);
	
	free(restructeredData);
	free(restructeredOutput);
	//cudaFree(gpu_x_values);
	cudaFreeArray(dataArray);
	
	return output;
}

template
short* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	float* echotimes, short* data, float threshold);

template
float* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	float* echotimes, short* data, float threshold);

template
unsigned short* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	float* echotimes, short* data, float threshold);

template
short* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	float* echotimes, float* data, float threshold);

template
float* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	float* echotimes, float* data, float threshold);

template
unsigned short* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	float* echotimes, float* data, float threshold);

template
short* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	float* echotimes, unsigned short* data, float threshold);

template
float* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	float* echotimes, unsigned short* data, float threshold);

template
unsigned short* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	float* echotimes, unsigned short* data, float threshold);