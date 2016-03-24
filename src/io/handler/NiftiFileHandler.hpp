/* 
 * File:   NiftiFileHandler.hpp
 * Author: christiantinauer
 *
 * Created on August 27, 2014, 1:20 PM
 */

#ifndef NIFTIFILEHANDLER_HPP
#define	NIFTIFILEHANDLER_HPP

#include "IFileHandler.hpp"

class NiftiFileHandler : public IFileHandler {
 public:
	NiftiFileHandler();
	
	~NiftiFileHandler();
 
	ProcessorData* Read(const string& filename, FileHandlerReadContext* context);
	
	void Write(const string& filename, ProcessorData* data, FileHandlerWriteContext* context);
};

#endif	/* NIFTIFILEHANDLER_HPP */