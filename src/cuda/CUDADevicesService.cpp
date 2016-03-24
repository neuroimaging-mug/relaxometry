/* 
 * File:   CUDADeviceHandler.cpp
 * Author: christiantinauer
 * 
 * Created on September 8, 2014, 2:10 PM
 */

#include "CUDADevicesService.hpp"

#include <stdexcept>
#include "cuda_runtime.h"
#include "../includes/StringExtensions.h"

CUDADevicesService::CUDADevicesService() {
}

vector<string> CUDADevicesService::listCUDADevices() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	vector<string> deviceInfos;
	for(int index = 0; index < deviceCount; ++index) {
		cudaDeviceProp* devProperties = new cudaDeviceProp();
		cudaGetDeviceProperties(devProperties, index);
		deviceInfos.push_back(
			to_string(index) + " - " + 
			string(devProperties->name));
		delete devProperties;
	}
	return deviceInfos;
}

void CUDADevicesService::setCUDADevice(int device) {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (device >= 0 && device < deviceCount)
		cudaSetDevice(device);
  else
		throw runtime_error("Device " + to_string<int>(device) + " does not exist. Device count: " + to_string<int>(deviceCount));
}
	
string CUDADevicesService::getCurrentCUDADevice() {
	int device;
	cudaGetDevice(&device);
	return CUDADevicesService::listCUDADevices()[device];
}

int CUDADevicesService::getSharedMemoryPerBlock() {
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp* devProperties = new cudaDeviceProp();
	cudaGetDeviceProperties(devProperties, device);
	return devProperties->sharedMemPerBlock;
}

int CUDADevicesService::getMaximumTexture2DHeight() {
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp* devProperties = new cudaDeviceProp();
	cudaGetDeviceProperties(devProperties, device);
	return devProperties->maxTexture2D[1];
}