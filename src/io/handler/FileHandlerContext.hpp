/* 
 * File:   FileHandlerReadContext.hpp
 * Author: christiantinauer
 *
 * Created on August 27, 2014, 12:34 PM
 */

#ifndef FILEHANDLERREADCONTEXT_HPP
#define	FILEHANDLERREADCONTEXT_HPP

#include <string>

using namespace std;

class FileHandlerReadContext {
 private:
	bool debug;
	
 public:
	FileHandlerReadContext(bool debug);
	
	virtual ~FileHandlerReadContext();

	bool getDebug();
};

class EchoesFileHandlerReadContext : public FileHandlerReadContext {
 private:
	int startIndex;
	int endIndex;
	float timeCorrectionFactor;
	
 public:
	EchoesFileHandlerReadContext(bool debug, int startIndex, int endIndex);
	
	virtual ~EchoesFileHandlerReadContext();
	
	int getStartIndex();
	
	int getEndIndex();
};

class TextFileHandlerReadContext : public FileHandlerReadContext {
 private:
	int expectedEntryCount;
	bool useDoublePrecision;
	
 public:
	TextFileHandlerReadContext(bool debug, int expectedEntryCount);
	
	~TextFileHandlerReadContext();

	int getExpectedEntryCount();
};

class FileHandlerWriteContext {
 private:
	bool debug;
	int startIndex;
	int endIndex;
	const string& metaDataSource;
	
 public:
	FileHandlerWriteContext(bool debug, int startIndex, int endIndex, const string& metaDataSource);
	
	virtual ~FileHandlerWriteContext();

	bool getDebug();
	
	int getStartIndex();
	
	int getEndIndex();
	
	const string& getMetaDataSource();
};

#endif	/* FILEHANDLERREADCONTEXT_HPP */