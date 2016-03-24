/* 
 * File:   DicomFileHandler.hpp
 * Author: christiantinauer
 *
 * Created on August 27, 2014, 1:15 PM
 */

#ifndef DICOMFILEHANDLER_HPP
#define	DICOMFILEHANDLER_HPP

#include "IFileHandler.hpp"

#include <gdcm-2.6/gdcmImageReader.h>

using namespace gdcm;

class DicomFileHandler : public IFileHandler {
 private:
	template<unsigned short Group, unsigned short Element> int GetAttributeValue(DataSet& ds);

 public:
	DicomFileHandler();

	~DicomFileHandler();

	ProcessorData* Read(const string& filename, FileHandlerReadContext* context);

	void Write(const string& filename, ProcessorData* data, FileHandlerWriteContext* context);
	
	void Anonymize(const string& filename);
};

#endif	/* DICOMFILEHANDLER_HPP */