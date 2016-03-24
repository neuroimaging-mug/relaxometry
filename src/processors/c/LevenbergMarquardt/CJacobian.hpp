/* 
 * File:   CJacobian.hpp
 * Author: christiantinauer
 *
 * Created on September 24, 2014, 3:45 PM
 */

#ifndef CJACOBIAN_HPP
#define	CJACOBIAN_HPP

template<typename NUMERICTYPE>
void CJacobian(
	short modelFunctionID,
	NUMERICTYPE* x_values, NUMERICTYPE* y_values, NUMERICTYPE* y_forward,
	NUMERICTYPE* result, int length,
	NUMERICTYPE* parameters, int parameters_length,
	NUMERICTYPE* constants, int constants_length);

#endif	/* CJACOBIAN_HPP */