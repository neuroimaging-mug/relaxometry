/* 
 * File:   StringExtensions.h
 * Author: christiantinauer
 *
 * Created on September 8, 2014, 4:14 PM
 */

#ifndef STRINGEXTENSIONS_H
#define	STRINGEXTENSIONS_H

#include <string>
#include <sstream>

using namespace std;

template <class T> inline string to_string(const T& t) {
	stringstream ss;
	ss << t;
	return ss.str();
}

#endif	/* STRINGEXTENSIONS_H */