/* 
 * File:   SimpleWeighting.cuh
 * Author: christiantinauer
 *
 * Created on September 30, 2015, 3:27 PM
 */

#ifndef SIMPLEWEIGHTING_CUH
#define	SIMPLEWEIGHTING_CUH

template<typename NTCALC>
__device__ inline NTCALC linearWeighting(NTCALC* weights, int length, int index) {
	return weights[index];
}

#endif	/* SIMPLEWEIGHTING_CUH */