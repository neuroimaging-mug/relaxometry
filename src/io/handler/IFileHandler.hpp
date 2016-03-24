/* 
 * File:   IFileHandler.hpp
 * Author: christiantinauer
 *
 * Created on August 27, 2014, 12:05 PM
 */

#ifndef IFILEHANDLER_HPP
#define	IFILEHANDLER_HPP

#include "FileHandlerContext.hpp"
#include "../data/ProcessorData.hpp"

#include <string>

using namespace std;

class IFileHandler {
 public:
	virtual ProcessorData* Read(const string& filename, FileHandlerReadContext* context) = 0;
	
	virtual void Write(const string& filename, ProcessorData* data, FileHandlerWriteContext* context) = 0;

	static IFileHandler* CreateFileHandlerFromFilenameExtension(const string& filename);
	
 private:
	static bool HasSuffix(const string& text, const string& suffix);
};

#endif	/* IFILEHANDLER_HPP */