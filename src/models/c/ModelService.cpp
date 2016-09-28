/* 
 * File:   ModelService.cpp
 * Author: christiantinauer
 * 
 * Created on September 20, 2014, 2:13 PM
 */

#include "ModelService.hpp"

#define _USE_MATH_DEFINES

#include <cmath>
#include <limits>

using namespace std;

//2 params, T1
template<typename NUMERICTYPE>
void ModelService::Exponential2ParametersR1(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	int index = constants[0];
	NUMERICTYPE* flipAngles = constants + 8;
	for(int i = 0; i < length; ++i)
		output[i] = parameters[1] * (1 - flipAngles[index] * exp(-input[i] * parameters[0]));
}

template<typename NUMERICTYPE>
void ModelService::Exponential2ParametersT1(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	int index = constants[0];
	NUMERICTYPE* flipAngles = constants + 8;
	for(int i = 0; i < length; ++i)
		output[i] = parameters[1] * (1 - flipAngles[index] * exp(-input[i] / parameters[0]));
}

//3 params, T1
template<typename NUMERICTYPE>
void ModelService::Exponential3ParametersR1(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	for(int i = 0; i < length; ++i)
		output[i] = parameters[1] * (1 - parameters[2] * exp(-input[i] * parameters[0]));
}

template<typename NUMERICTYPE>
void ModelService::Exponential3ParametersT1(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	for(int i = 0; i < length; ++i)
		output[i] = parameters[1] * (1 - parameters[2] * exp(-input[i] / parameters[0]));
}

//T2
template<typename NUMERICTYPE>
void ModelService::ExponentialR2(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	for(int i = 0; i < length; ++i)
		output[i] = parameters[1] * exp(-input[i] * parameters[0]);
}

template<typename NUMERICTYPE>
void ModelService::ExponentialT2(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	for(int i = 0; i < length; ++i)
		output[i] = parameters[1] * exp(-input[i] / parameters[0]);
}

//LukzenSavelov
template<typename NUMERICTYPE>
complex<NUMERICTYPE> ZFunction(complex<NUMERICTYPE> z, NUMERICTYPE m0, NUMERICTYPE t1,
	NUMERICTYPE t2, NUMERICTYPE tau, NUMERICTYPE flipAngle,
	int profileLength) {
	NUMERICTYPE kappa1 = exp(-tau / t1);
	NUMERICTYPE kappa2 = exp(-tau / t2);
	complex<NUMERICTYPE> last = z * z * kappa1 * kappa2;
	complex<NUMERICTYPE> zKappa2 = z * kappa2;
	
	NUMERICTYPE profile[] = {1.0000, 0.9608, 0.8521, 0.6977, 0.5273, 0.3679, 0.2369, 0.1409, 0.0773, 0.0392, 0.0183, 0.0079, 0.0032};
	
	complex<NUMERICTYPE> profileSum = 0;
	for(int p = 0; p < profileLength; ++p) {
		complex<NUMERICTYPE> zCosAlpha = z * cos(flipAngle * profile[p]);

		complex<NUMERICTYPE> numerator = (((NUMERICTYPE)1) + zKappa2) * (((NUMERICTYPE)1) - zCosAlpha * (kappa1 + kappa2) + last);
		complex<NUMERICTYPE> denominator = (-((NUMERICTYPE)1) + zKappa2) * (-((NUMERICTYPE)1) + zCosAlpha * (kappa1 - kappa2) + last);

		profileSum += (m0 * ((NUMERICTYPE)0.5) * (((NUMERICTYPE)1) + sqrt(numerator / denominator)));
	}
	return profileSum / ((NUMERICTYPE)profileLength);
}

template<typename NUMERICTYPE>
void ModelService::LukzenSavelov(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	
	int index = (int)constants[0];
	NUMERICTYPE tau = constants[4];
  int profileLength = (int)constants[5];
	int dataLength = (int)constants[6];
  int m = (int)constants[7];
  NUMERICTYPE flipAngle = constants[8 + index] * M_PI;
  NUMERICTYPE t1 = constants[8 + dataLength + index];
  
  complex<NUMERICTYPE> f[m];
  for(int i = 0; i < m; ++i) {
    NUMERICTYPE theta = i * 2 * M_PI / m;
    complex<NUMERICTYPE> z = exp(complex<NUMERICTYPE>(0, theta));
    f[i] = ZFunction(z, parameters[1], t1, parameters[0], tau, flipAngle, 
			profileLength);
  }

  ForwardFFT(f, m);
	
	for(int i = 0; i < length; ++i) {
		int j = i + 1; // skip point at zero
		output[i] = abs(f[j]) / m;
	}
}

//Inplace version of rearrange function
template<typename NUMERICTYPE>
void Rearrange(complex<NUMERICTYPE>* data, int length) {
	// Swap position
	unsigned int target = 0;
	// Process all positions of input signal
	for(unsigned int index = 0; index < length; ++index) {
		// Only for not yet swapped entries
		if(target > index) {
			// Swap entries
			complex<NUMERICTYPE> temp = data[target];
			data[target] = data[index];
			data[index] = temp;
		}
		// Bit mask
		unsigned int mask = length;
		// While bit is set
		while(target & (mask >>= 1)) {
			// Drop bit
			target &= ~mask;
		}
		// The current bit is 0 - set it
		target |= mask;
	}
}

template<typename NUMERICTYPE>
void Perform(complex<NUMERICTYPE>* data, int length) {
	// Iteration through dyads, quadruples, octads and so on...
	for(unsigned int stage = 1; stage < length; stage <<= 1) {
		// jump to the next entry of the same transform factor
		const unsigned int jump = stage << 1;
		// Angle increment
		const double delta = -M_PI / stage;
		// Auxiliary sin(delta / 2)
		const double sine = sin(delta * .5);
		// multiplier for trigonometric recurrence
		const complex<NUMERICTYPE> multiplier(-2. * sine * sine, sin(delta));
		// Start value for transform factor, fi = 0
		complex<NUMERICTYPE> factor(1.);
		//Iteration through groups of different transform factor
		for(unsigned int group = 0; group < stage; ++group) {
			// Iteration within group 
			for(unsigned int pair = group; pair < length; pair += jump) {
				// Match position
				const unsigned int match = pair + stage;
				// Second term of two-point transform
				const complex<NUMERICTYPE> product(factor * data[match]);
				//   Transform for fi + pi
				data[match] = data[pair] - product;
				//   Transform for fi
				data[pair] += product;
			}
			//   Successive transform factor via trigonometric recurrence
			factor = multiplier * factor + factor;
		}
	}
}

template<typename NUMERICTYPE>
void ModelService::ForwardFFT(complex<NUMERICTYPE>* data, int length) {
	Rearrange(data, length);
	Perform(data, length);
}

//LinearRegression
template<typename NUMERICTYPE>
void ModelService::LinearRegressionT2(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	NUMERICTYPE ySum = 0;
	for(int index = 0; index < length; ++index)
		ySum += output[index];
	NUMERICTYPE yMean = log(ySum / (NUMERICTYPE)length);
	
	NUMERICTYPE meanTime = (input[0] + input[length - 1]) / 2;
	NUMERICTYPE covariance = 0; // SS_xy
	NUMERICTYPE variance = 0; // SS_xx
	for(int index = 0; index < length; ++index) {
		NUMERICTYPE y = output[index];
		NUMERICTYPE yis = log(y);
		NUMERICTYPE xis = input[index];
		NUMERICTYPE xDiff = xis - meanTime;
		covariance += xDiff * (yis - yMean);
		variance += xDiff * xDiff;
	}

	NUMERICTYPE k = covariance / variance;
	
	//p1 (T2)
	NUMERICTYPE p1 = k == 0 ? 0 : 1 / -k;
	parameters[0] = p1;
	
	//p2 (M0)
	NUMERICTYPE d = yMean - k * meanTime;
	NUMERICTYPE p2 = exp(d);
	parameters[1] = (NUMERICTYPE)p2;
}

template<typename NUMERICTYPE>
void ModelService::LinearRegressionR2(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	LinearRegressionT2(input, output, length, 
		parameters, parametersLength,
		constants, constantsLength);
	parameters[0] = 1 / parameters[0];
}

template<typename NUMERICTYPE>
void ModelService::LinearRegressionT1(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	NUMERICTYPE move = output[length - 1] * 1.01;
	if(move < 100) {
		parameters[0] = 0;
		parameters[1] = 0;
		return;
	}
	
	move = 2000;
	
	NUMERICTYPE t1 = 1200;// constants[1];
	NUMERICTYPE a = 1.8;//constants[3];
	NUMERICTYPE deltaT = 300;
	NUMERICTYPE extremumAt = log(a)*t1;
	
	NUMERICTYPE xSum = 0;
	NUMERICTYPE ySum = 0;
	for(int index = 0; index < length; ++index) {
		xSum += input[index];
		ySum += output[index] * -1 + move;
	}
	NUMERICTYPE xMean = xSum / (NUMERICTYPE)length;
	NUMERICTYPE yMean = log(ySum / (NUMERICTYPE)length);
	
	NUMERICTYPE covariance = 0; // SS_xy
	NUMERICTYPE variance = 0; // SS_xx
	for(int index = 0; index < length; ++index) {
		NUMERICTYPE xis = input[index];
		NUMERICTYPE y = output[index] * -1 + move;
		NUMERICTYPE yis = log(y);
		NUMERICTYPE xDiff = xis - xMean;
		covariance += xDiff * (yis - yMean);
		variance += xDiff * xDiff;
	}

	NUMERICTYPE k = covariance / variance;
  
	//p1 (T1)
	NUMERICTYPE p1 = k == 0 ? 0 : 1 / (-2 * k);
	parameters[0] = p1;
	
	//p2 (M0)
	NUMERICTYPE d = yMean - k * xMean;
	NUMERICTYPE p2 = exp(d);
	parameters[1] = p2;
	
	if(parametersLength == 3) {
		//p3 (a)
		parameters[2] = a;
	}
}

template<typename NUMERICTYPE>
void ModelService::LinearRegressionR1(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	LinearRegressionT1(input, output, length, 
		parameters, parametersLength,
		constants, constantsLength);
	parameters[0] = 1 / parameters[0];
}

//Simple start values
template<typename NUMERICTYPE>
void ModelService::SimpleStartValues(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	for(int index = 0; index < parametersLength; ++index)
		parameters[index] = constants[1 + index];
}


template<typename NUMERICTYPE>
NUMERICTYPE ModelService::LinearWeighting(
	NUMERICTYPE* weights, int length, int index) {
	return weights[index];
}

template<typename NUMERICTYPE>
NUMERICTYPE Min(NUMERICTYPE* values, int length) {
	NUMERICTYPE min = numeric_limits<NUMERICTYPE>::max();
	for(int index = 0; index < length; ++index) {
		NUMERICTYPE value = values[index];
		if(value < min)
			min = value;
	}
	return min;
}

template<typename NUMERICTYPE>
NUMERICTYPE ModelService::InverseMinimumWeighting(
	NUMERICTYPE* weights, int length, int index) {
	return Min(weights, length) / weights[index];
}
	
template<typename NUMERICTYPE>
NUMERICTYPE ModelService::InverseQuadraticWeighting(
	NUMERICTYPE* weights, int length, int index) {
	NUMERICTYPE weight = weights[index];
	return 1 / (weight * weight);
}


//double
template
void ModelService::Exponential2ParametersR1(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::Exponential2ParametersT1(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::Exponential3ParametersR1(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::Exponential3ParametersT1(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::ExponentialR2(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::ExponentialT2(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::LukzenSavelov(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::LinearRegressionR1(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::LinearRegressionT1(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::LinearRegressionR2(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::LinearRegressionT2(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);

template
void ModelService::SimpleStartValues(
	double* input, double* output, int length,
	double* parameters, int parametersLength,
	double* constants, int constantsLength);


template
double ModelService::LinearWeighting(
	double* weights, int length, int index);

template
double ModelService::InverseMinimumWeighting(
	double* weights, int length, int index);
	
template
double ModelService::InverseQuadraticWeighting(
	double* weights, int length, int index);


//float
template
void ModelService::Exponential2ParametersR1(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::Exponential2ParametersT1(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::Exponential3ParametersR1(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::Exponential3ParametersT1(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::ExponentialR2(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::ExponentialT2(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::LukzenSavelov(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::LinearRegressionR1(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::LinearRegressionT1(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::LinearRegressionR2(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::LinearRegressionT2(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);

template
void ModelService::SimpleStartValues(
	float* input, float* output, int length,
	float* parameters, int parametersLength,
	float* constants, int constantsLength);


template
float ModelService::LinearWeighting(
	float* weights, int length, int index);

template
float ModelService::InverseMinimumWeighting(
	float* weights, int length, int index);
	
template
float ModelService::InverseQuadraticWeighting(
	float* weights, int length, int index);