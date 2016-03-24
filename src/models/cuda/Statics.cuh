/* 
 * File:   CudaTexture.cuh
 * Author: christiantinauer
 *
 * Created on August 10, 2015, 3:42 PM
 */

#ifndef CUDATEXTURE_CUH
#define	CUDATEXTURE_CUH

static __constant__ float processConstants[263];

static texture<float, 2, cudaReadModeElementType> floatTexture;

#endif	/* CUDATEXTURE_CUH */