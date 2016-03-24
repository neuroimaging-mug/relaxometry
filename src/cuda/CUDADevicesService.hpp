/* 
 * File:   CUDADeviceHandler.hpp
 * Author: christiantinauer
 *
 * Created on September 8, 2014, 2:10 PM
 */

#ifndef CUDADEVICEHANDLER_HPP
#define	CUDADEVICEHANDLER_HPP

#include <string>
#include <vector>

using namespace std;

class CUDADevicesService {
 private:
	CUDADevicesService();
	
 public:
	static vector<string> listCUDADevices();

	static void setCUDADevice(int device);
	
	static string getCurrentCUDADevice();
	
	static int getSharedMemoryPerBlock();
	
	static int getMaximumTexture2DHeight(); 
};

#endif	/* CUDADEVICEHANDLER_HPP */