/* 
 * File:   ModelService.hpp
 * Author: christiantinauer
 *
 * Created on September 20, 2014, 2:13 PM
 */

#ifndef MODELSERVICE_HPP
#define	MODELSERVICE_HPP

#include <complex>

using namespace std;

class ModelService {
 private:
	ModelService();
	
	template<typename NUMERICTYPE>
	static void ForwardFFT(complex<NUMERICTYPE>* data, int length);

 public: 
 	template<typename NUMERICTYPE>
	static void Exponential2ParametersR1(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
 
	template<typename NUMERICTYPE>
	static void Exponential2ParametersT1(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
	
	 template<typename NUMERICTYPE>
	static void Exponential3ParametersR1(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
 
	template<typename NUMERICTYPE>
	static void Exponential3ParametersT1(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
	
	template<typename NUMERICTYPE>
	static void ExponentialR2(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
	
	template<typename NUMERICTYPE>
	static void ExponentialT2(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
	
	template<typename NUMERICTYPE>
	static void LukzenSavelov(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
	
        template<typename NUMERICTYPE>
	static void LinearRegressionR1(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
	
	template<typename NUMERICTYPE>
	static void LinearRegressionT1(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
        
	template<typename NUMERICTYPE>
	static void LinearRegressionR2(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
	
	template<typename NUMERICTYPE>
	static void LinearRegressionT2(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);

	
	template<typename NUMERICTYPE>
	static void SimpleStartValues(
		NUMERICTYPE* input, NUMERICTYPE* output, int length,
		NUMERICTYPE* parameters, int parametersLength,
		NUMERICTYPE* constants, int constantsLength);
	
	template<typename NUMERICTYPE>
	static NUMERICTYPE LinearWeighting(
		NUMERICTYPE* weights, int length, int index);
	
	template<typename NUMERICTYPE>
	static NUMERICTYPE InverseMinimumWeighting(
		NUMERICTYPE* weights, int length, int index);
	
	template<typename NUMERICTYPE>
	static NUMERICTYPE InverseQuadraticWeighting(
		NUMERICTYPE* weights, int length, int index);
};

#endif	/* MODELSERVICE_HPP */