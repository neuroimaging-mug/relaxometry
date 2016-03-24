/* 
 * File:   MatlabHandler.hpp
 * Author: christiantinauer
 *
 * Created on August 29, 2014, 4:28 PM
 */

#ifndef MATLABFILEHANDLER_HPP
#define	MATLABFILEHANDLER_HPP

#include "IFileHandler.hpp"

class TextFileHandler : public IFileHandler {
 public:
	TextFileHandler();
	
	~TextFileHandler();
 
	ProcessorData* Read(const string& filename, FileHandlerReadContext* context);
	
	void Write(const string& filename, ProcessorData* data, FileHandlerWriteContext* context);
};

#endif	/* MATLABFILEHANDLER_HPP */