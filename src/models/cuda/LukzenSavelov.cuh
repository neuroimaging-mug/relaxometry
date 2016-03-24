/* 
 * File:   LukzenSavelov.cuh
 * Author: christiantinauer
 *
 * Created on January 19, 2015, 9:25 PM
 */

#ifndef LUKZENSAVELOV_CUH
#define	LUKZENSAVELOV_CUH

#include "Complex.cuh"
#include "Statics.cuh"

//Inplace version of rearrange function
__device__ inline void CudaRearrange(c* data, int length) {
	// Swap position
	unsigned int target = 0;
	// Process all positions of input signal
	for(unsigned int index = 0; index < length; ++index) {
		// Only for not yet swapped entries
		if(target > index) {
			// Swap entries
			c temp = data[target];
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

__device__ inline void CudaPerform(c* data, int length) {
	const float pi = 3.14159265358979323846f;

	// Iteration through dyads, quadruples, octads and so on...
	for(unsigned int stage = 1; stage < length; stage <<= 1) {
		// jump to the next entry of the same transform factor
		const unsigned int jump = stage << 1;
		// Angle increment
		const double delta = -pi / stage;
		// Auxiliary sin(delta / 2)
		const double sine = sin(delta * .5);
		// multiplier for trigonometric recurrence
		const c multiplier = make_cuComplex(-2.0f * sine * sine, sin(delta));
		// Start value for transform factor, fi = 0
		c factor = make_cuComplex(1.0f, 0.0f);
		//Iteration through groups of different transform factor
		for(unsigned int group = 0; group < stage; ++group) {
			// Iteration within group 
			for(unsigned int pair = group; pair < length; pair += jump) {
				// Match position
				const unsigned int match = pair + stage;
				// Second term of two-point transform
				const c product(factor * data[match]);
				//   Transform for fi + pi
				data[match] = data[pair] - product;
				//   Transform for fi
				data[pair] = data[pair] + product;
			}
			//   Successive transform factor via trigonometric recurrence
			factor = multiplier * factor + factor;
		}
	}
}

__device__ inline  void CudaForwardFFT(c* data, int length) {
	CudaRearrange(data, length);
	CudaPerform(data, length);
}

__device__ inline c CudaZFunction(c z, float m0, float t1, float t2, float tau, float alpha, int profileLength) {
	float kappa1 = expf(-tau / t1);
	float kappa2 = expf(-tau / t2);
	c last = z * z * make_cuComplex(kappa1 * kappa2, 0.0f);
	c zKappa2 = z * make_cuComplex(kappa2, 0.0f);
	
	float profile[] = { 1.0000, 0.9608, 0.8521, 0.6977, 0.5273, 0.3679, 0.2369, 0.1409, 0.0773, 0.0392, 0.0183, 0.0079, 0.0032 };
	c sum = make_cuComplex(0.f, 0.f);
	c zCosAlpha, numerator, denominator, quotient, squareRoot;
	
	for(int p = 0; p < profileLength; ++p) {
		zCosAlpha = z * make_cuComplex(cos(alpha * profile[p]), 0.0f);
		numerator = (make_cuComplex(1.0f, 0.0f) + zKappa2) * (make_cuComplex(1.0f, 0.0f) - zCosAlpha * make_cuComplex(kappa1 + kappa2, 0.0f) + last);
	  denominator = (make_cuComplex(-1.0f, 0.0f) + zKappa2) * (make_cuComplex(-1.0f, 0.0f) + zCosAlpha * make_cuComplex(kappa1 - kappa2, 0.0f) + last);
		quotient = numerator / denominator;
		squareRoot = sqrt(quotient);
		sum = sum + make_cuComplex(m0 * 0.5f, 0.0f) * (make_cuComplex(1.0f, 0.0f) + squareRoot);
	}
	
	return sum / make_cuComplex((float)profileLength, 0.0f);
}

template<typename NUMERICTYPE>
__device__ inline void lukzenSavelov(
	NUMERICTYPE* input, NUMERICTYPE* output, int length,
	NUMERICTYPE* parameters, int parametersLength,
	NUMERICTYPE* constants, int constantsLength) {
	const float PI = 3.141592653589793238462f;
	
	const float tau = processConstants[3];
	const int profileLength = (int)processConstants[4];
  const int m = (int)processConstants[6];
	int index = blockDim.x * blockIdx.x + threadIdx.x;
  float flipAngle = tex2D(floatTexture, length + 1, index) * PI;
	float t1 = tex2D(floatTexture, length + 2, index);
	
	c f[128];
	for(int i = 0; i < m; ++i) {
    float theta = i * 2 * PI / m;
    c z = make_cuComplex(cos(theta), sin(theta));
		f[i] = CudaZFunction(z, (float)parameters[1], t1, (float)parameters[0], tau, flipAngle, profileLength);
	}

  CudaForwardFFT(f, m);
	
	for(int i = 0; i < length; ++i) {
		int j = i + 1; // skip point at zero
		output[i] = abs(f[j]) / m;
	}
}

#endif	/* LUKZENSAVELOV_CUH */