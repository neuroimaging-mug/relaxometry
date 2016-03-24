/* 
 * File:   T2AlphaWeighting.cuh
 * Author: christiantinauer
 *
 * Created on September 30, 2015, 3:27 PM
 */

#ifndef T2ALPHAWEIGHTING_CUH
#define	T2ALPHAWEIGHTING_CUH

template<typename NTCALC>
__device__ inline NTCALC inverseQuadraticWeighting(NTCALC* weights, int length, int index) {
	NTCALC weight = weights[index];
	return 1. / (weight * weight);
}

#endif	/* T2ALPHAWEIGHTING_CUH */