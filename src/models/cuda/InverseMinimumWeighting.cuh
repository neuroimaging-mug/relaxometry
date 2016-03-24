/* 
 * File:   T2ErrorWeighting.cuh
 * Author: christiantinauer
 *
 * Created on September 30, 2015, 3:27 PM
 */

#ifndef T2ERRORWEIGHTING_CUH
#define	T2ERRORWEIGHTING_CUH

template<typename NTCALC>
__device__ inline NTCALC cudaMin(NTCALC* values, int length) {
	NTCALC min = 1;
	for(int index = 0; index < length; ++index) {
		NTCALC value = values[index];
		if(value < min)
			min = value;
	}
	return min;
}

template<typename NTCALC>
__device__ inline NTCALC inverseMinimumWeighting(NTCALC* weights, int length, int index) {
	return cudaMin(weights, length) / weights[index];
}

#endif	/* T2ERRORWEIGHTING_CUH */