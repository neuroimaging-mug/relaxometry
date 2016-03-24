/* 
 * File:   CudaCalculateInverseMatrix.cuh
 * Author: christiantinauer
 *
 * Created on September 29, 2015, 4:31 PM
 */

#ifndef CUDACALCULATEINVERSEMATRIX_CUH
#define	CUDACALCULATEINVERSEMATRIX_CUH

template<typename NTCALC>
__device__ void CudaCalculateInverseMatrix(NTCALC* matrix, NTCALC* inverseMatrix, int parametersCount) {
	if(parametersCount == 2) {
		NTCALC det = 1 / (matrix[0] * matrix[3] - matrix[1] * matrix[2]);
		inverseMatrix[0] = matrix[3] * det;
		inverseMatrix[1] = -matrix[1] * det;
		inverseMatrix[2] = -matrix[2] * det;
		inverseMatrix[3] = matrix[0] * det;
	} else if(parametersCount == 3) {
		NTCALC det = 1 / (
			(	matrix[0] * matrix[4] * matrix[8] +
				matrix[1] * matrix[5] * matrix[6] +
				matrix[2] * matrix[3] * matrix[7]) -
			(	matrix[2] * matrix[4] * matrix[6] +
				matrix[0] * matrix[5] * matrix[7] +
				matrix[1] * matrix[3] * matrix[8]));
		inverseMatrix[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * det;
		inverseMatrix[1] = -(matrix[1] * matrix[8] - matrix[2] * matrix[7]) * det;
		inverseMatrix[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * det;
		inverseMatrix[3] = -(matrix[3] * matrix[8] - matrix[5] * matrix[6]) * det;
		inverseMatrix[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * det;
		inverseMatrix[5] = -(matrix[0] * matrix[5] - matrix[2] * matrix[3]) * det;
		inverseMatrix[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * det;
		inverseMatrix[7] = -(matrix[0] * matrix[7] - matrix[1] * matrix[6]) * det;
		inverseMatrix[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * det;
	}
}

#endif	/* CUDACALCULATEINVERSEMATRIX_CUH */