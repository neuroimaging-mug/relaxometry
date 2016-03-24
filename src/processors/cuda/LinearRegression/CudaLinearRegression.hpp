/* 
 * File:   CudaLinearRegression.hpp
 * Author: christiantinauer
 *
 * Created on September 12, 2014, 5:24 PM
 */

#ifndef CUDALINEARREGRESSION_HPP
#define	CUDALINEARREGRESSION_HPP

template<typename NTINPUT, typename NTOUTPUT, typename NTCALC>
NTOUTPUT* CudaProcessLinearRegression(
	short modelFunctionID,
	int startIndex, int endIndex, int columnCount, int rowCount, int sliceCount,
	NTCALC* echotimes, NTINPUT* data, NTCALC threshold);

#endif	/* CUDALINEARREGRESSION_HPP */