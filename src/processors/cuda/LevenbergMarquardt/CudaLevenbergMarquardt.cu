#include "../../../cuda/CUDADevicesService.hpp"
#include "../../../datatransfer/DataTransfer.cuh"
#include "../LevenbergMarquardtFletcher/CudaStepVariables.cuh"
#include "../LevenbergMarquardtFletcher/CudaCalculateAdjustedRsquareValue.cuh"
#include "../../../models/cuda/Statics.cuh"
#include "../../../models/cuda/Model.cuh"
#include "../ErrorHandling.cuh"
#include "../MemoryHandling.cuh"

#include <cuda_runtime.h>

#include "CudaLevenbergMarquardtCore.cu"

template<typename NTOUTPUT, typename NTCALC>
__global__ void CudaLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int columnCount, int rowCount, int echoesCount,
	int sharedMemoryPerThread,
	NTCALC* weights, NTCALC* x_values, NTCALC* artifacts,
	NTOUTPUT* output,
	NTCALC EPSILON_1, NTCALC EPSILON_2) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index >= rowCount)
    return;

#ifdef USE_SHARED_MEMORY
	extern __shared__ NTCALC sharedMemory[];
	NTCALC* threadSharedMemory = sharedMemory + threadIdx.x * sharedMemoryPerThread;
	NTCALC* parameters = threadSharedMemory;
#else
	NTCALC* parameters = artifacts + index * sharedMemoryPerThread;
#endif
	
	CudaStepVariables<NTCALC> stepVariables;
	stepVariables.y_hat = parameters + parametersCount;
	stepVariables.dydp = stepVariables.y_hat + echoesCount;
	stepVariables.alpha = stepVariables.dydp + echoesCount * parametersCount;
  stepVariables.beta =  stepVariables.alpha + parametersCount * parametersCount;
	
	//parameters start value
	cudaModelFunction(
		startModelFunctionID,
		x_values, (NTCALC*)NULL, echoesCount,
		parameters, parametersCount,
		(NTCALC*)NULL, 0);
		
	__syncthreads();
	
	NTCALC parameters_min[3];
	NTCALC parameters_max[3];
	for(int index = 0; index < parametersCount; ++index) {
		parameters_min[index] = processConstants[7 + index * 2];
		parameters_max[index] = processConstants[7 + index * 2 + 1];
	}
	
	NTCALC parameters_try[3];
	NTCALC delta_p[3];
	
	int sliceIndex = (int)tex2D(floatTexture, echoesCount, index);
	
	CudaLevenbergMarquardtCore(
		modelFunctionID,
		residualWeightingFunctionID, alphaWeightingFunctionID,
		columnCount, rowCount, echoesCount,
		weights + sliceIndex * echoesCount, x_values, stepVariables.beta + parametersCount,
		parameters, parametersCount,
		parameters_min, parameters_max, delta_p, parameters_try, &stepVariables,
		EPSILON_1, EPSILON_2);

	__syncthreads();
	
	//p1 (T1/T2)
	output[index * (parametersCount + 1)] = (NTOUTPUT)parameters[0];
	
	//p2 (M0)
	output[index * (parametersCount + 1) + 1] = (NTOUTPUT)parameters[1];
	
	//p3 (FA)
	if(parametersCount == 3)
		output[index * (parametersCount + 1) + 2] = (NTOUTPUT)parameters[2];
	
	//GOF
	bool calc = false;
	for(int index = 0; index < parametersCount; ++index)
		if(parameters[index] != 1) {
			calc = true;
			break;
		}
	
	output[index * (parametersCount + 1) + parametersCount] = calc 
		? (NTOUTPUT)CudaCalculateAdjustedRsquareValue(
				stepVariables.y_hat, echoesCount, parametersCount)
		: 0;
}

template<typename NTINPUT, typename NTOUTPUT, typename NTCALC>
NTOUTPUT* CudaProcessLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount, 
	int sliceCount, NTCALC* echotimes, NTCALC* weights, NTINPUT* data,
	NTCALC threshold, NTCALC* constants, int constantsLength,
	NTCALC* parameterBoundaries,
	bool needsFlipAnglesData, bool needsT1Data, int threadCount,
	NTCALC EPSILON_1, NTCALC EPSILON_2) {
	int sliceSize = columnCount * rowCount;
	int dataLength = sliceSize * sliceCount;
	int echoesCount = endIndex - startIndex + 1;
	
	//mark valid data
	bool validData[dataLength];
	int validDataLength = MarkValidData(data, validData, dataLength, threshold);
	
	int restructeredColumnsCount = echoesCount + 1; //for sliceIndex
	if(needsFlipAnglesData)
		++restructeredColumnsCount;
	if(needsT1Data)
		++restructeredColumnsCount;
	
	//restructure data
	NTCALC* restructeredData = new NTCALC[validDataLength * restructeredColumnsCount];
	RestructureDataForward(
		data, 
		needsFlipAnglesData ? constants + 7 : (float*)NULL,
		needsT1Data ? constants + 7 + sliceSize : (float*)NULL,
		validData, dataLength, sliceSize, echoesCount, restructeredData);
	
	int maxTexture2DHeight = CUDADevicesService::getMaximumTexture2DHeight(); 
	int kernelCalls = validDataLength / maxTexture2DHeight;
	if(validDataLength % maxTexture2DHeight != 0)
		++kernelCalls;
	int restructeredRowsCount = min(maxTexture2DHeight, validDataLength);

	//processConstants
	float constBuilder[263] = {};
	for(int index = 0; index < 7; ++index)
		constBuilder[index] = constants[index];
	for(int index = 0; index < parametersCount * 2; ++index)
		constBuilder[index + 7] = parameterBoundaries[index];
	for(int index = 0; index < echoesCount; ++index)
		constBuilder[index + 13] = echotimes[index];
	HANDLE_CUDA_ERROR(cudaMemcpyToSymbol(processConstants, constBuilder,
		263 * sizeof(float), 0, cudaMemcpyHostToDevice));
	
	NTCALC* gpu_x_values = NULL; //processConstants + 13; //AllocAndCopyToDevice(echotimes, echoesCount);
	NTCALC* gpu_weights = AllocAndCopyToDevice(weights, echoesCount * sliceCount);
	
	NTOUTPUT* gpu_output = AllocOnDevice<NTOUTPUT>(validDataLength * (parametersCount + 1));
	
	//CudaArray + Texture for restructered data
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<NTCALC>();
	
	cudaExtent restructeredDataExtent = 
		make_cudaExtent(restructeredColumnsCount, restructeredRowsCount, 0);
	
	cudaArray* dataArray = NULL;
	HANDLE_CUDA_ERROR(cudaMalloc3DArray(
		&dataArray, &channelFormatDesc, restructeredDataExtent, cudaArrayDefault));

	//TODO on better gpus -> for now global memory
	//Calc needed shared memory and matching threads/blocks
	int sharedMemoryPerThread =
		parametersCount +											//params
		echoesCount +													//y_hat
		echoesCount * parametersCount +				//dydp
		parametersCount * parametersCount +		//alpha
		parametersCount +											//beta
		echoesCount;													//temp

#ifdef USE_SHARED_MEMORY
	int sharedMemoryPerThreadBytes =
		sharedMemoryPerThread * sizeof(NTCALC);
	int availableSharedMemoryPerBlock =
		CUDADevicesService::getSharedMemoryPerBlock();
	int threadsPerBlock = min((int)(availableSharedMemoryPerBlock / sharedMemoryPerThreadBytes), int threadCount);
#else
	NTCALC* gpu_artifacts =
		AllocOnDevice<NTCALC>(sharedMemoryPerThread * restructeredRowsCount);
	int threadsPerBlock = threadCount;
#endif
		
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
		
#ifdef USE_SHARED_MEMORY
		CudaLevenbergMarquardt<<<blockCount, threadsPerBlock, sharedMemoryPerThreadBytes>>>(
			startModelFunctionID, modelFunctionID,
			residualWeightingFunctionID, alphaWeightingFunctionID,
			parametersCount,
			restructeredColumnsCount, restructeredRowsCount, echoesCount,
			sharedMemoryPerThread,
			gpu_weights, gpu_x_values, (NTCALC*)NULL,
			gpu_output + kernelIndex * restructeredRowsCount * (parametersCount + 1),
			EPSILON_1, EPSILON_2);
#else
		CudaLevenbergMarquardt<<<blockCount, threadsPerBlock>>>(
			startModelFunctionID, modelFunctionID,
			residualWeightingFunctionID, alphaWeightingFunctionID,
			parametersCount,
			restructeredColumnsCount, restructeredRowsCount, echoesCount,
			sharedMemoryPerThread,
			gpu_weights, gpu_x_values, gpu_artifacts,
			gpu_output + kernelIndex * restructeredRowsCount * (parametersCount + 1),
			EPSILON_1, EPSILON_2);
#endif
		
		HANDLE_CUDA_ERROR(cudaGetLastError());
		HANDLE_CUDA_ERROR(cudaUnbindTexture(floatTexture));
	}
	
	NTOUTPUT* output = new NTOUTPUT[dataLength * (parametersCount + 1)];
	NTOUTPUT* restructeredOutput = CopyFromDeviceAndFree(gpu_output, 
		validDataLength * (parametersCount + 1));
	HANDLE_CUDA_ERROR(cudaGetLastError());
	RestructureDataBackward(output, validData, dataLength, restructeredOutput,
		parametersCount, true);
	
	free(restructeredData);
	free(restructeredOutput);
#ifndef USE_SHARED_MEMORY
	cudaFree(gpu_artifacts);
#endif
	cudaFree(gpu_weights);
	//cudaFree(gpu_x_values);
	cudaFreeArray(dataArray);
	
	return output;
}

template
short* CudaProcessLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount, 
	int sliceCount, float* echotimes, float* weights, short* data,
	float threshold, float* constants, int constantsLength,
	float* parameterBoundaries,
	bool needsFlipAnglesData, bool needsT1Data, int threadCount,
	float EPSILON_1, float EPSILON_2);

template
float* CudaProcessLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount, 
	int sliceCount, float* echotimes, float* weights, short* data,
	float threshold, float* constants, int constantsLength,
	float* parameterBoundaries,
	bool needsFlipAnglesData, bool needsT1Data, int threadCount,
	float EPSILON_1, float EPSILON_2);

template
unsigned short* CudaProcessLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount, 
	int sliceCount, float* echotimes, float* weights, short* data,
	float threshold, float* constants, int constantsLength,
	float* parameterBoundaries,
	bool needsFlipAnglesData, bool needsT1Data, int threadCount,
	float EPSILON_1, float EPSILON_2);

template
short* CudaProcessLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount, 
	int sliceCount, float* echotimes, float* weights, float* data,
	float threshold, float* constants, int constantsLength,
	float* parameterBoundaries,
	bool needsFlipAnglesData, bool needsT1Data, int threadCount,
	float EPSILON_1, float EPSILON_2);

template
float* CudaProcessLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount, 
	int sliceCount, float* echotimes, float* weights, float* data,
	float threshold, float* constants, int constantsLength,
	float* parameterBoundaries,
	bool needsFlipAnglesData, bool needsT1Data, int threadCount,
	float EPSILON_1, float EPSILON_2);

template
unsigned short* CudaProcessLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount, 
	int sliceCount, float* echotimes, float* weights, float* data,
	float threshold, float* constants, int constantsLength,
	float* parameterBoundaries,
	bool needsFlipAnglesData, bool needsT1Data, int threadCount,
	float EPSILON_1, float EPSILON_2);

template
short* CudaProcessLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount, 
	int sliceCount, float* echotimes, float* weights, unsigned short* data,
	float threshold, float* constants, int constantsLength,
	float* parameterBoundaries,
	bool needsFlipAnglesData, bool needsT1Data, int threadCount,
	float EPSILON_1, float EPSILON_2);

template
float* CudaProcessLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount, 
	int sliceCount, float* echotimes, float* weights, unsigned short* data,
	float threshold, float* constants, int constantsLength,
	float* parameterBoundaries,
	bool needsFlipAnglesData, bool needsT1Data, int threadCount,
	float EPSILON_1, float EPSILON_2);

template
unsigned short* CudaProcessLevenbergMarquardt(
	short startModelFunctionID, short modelFunctionID,
	short residualWeightingFunctionID, short alphaWeightingFunctionID,
	int parametersCount,
	int startIndex, int endIndex, int columnCount, int rowCount, 
	int sliceCount, float* echotimes, float* weights, unsigned short* data,
	float threshold, float* constants, int constantsLength,
	float* parameterBoundaries,
	bool needsFlipAnglesData, bool needsT1Data, int threadCount,
	float EPSILON_1, float EPSILON_2);