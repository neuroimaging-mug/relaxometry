/* 
 * File:   FileHandlerReadContext.cpp
 * Author: christiantinauer
 * 
 * Created on August 27, 2014, 12:34 PM
 */

#include "FileHandlerContext.hpp"

FileHandlerReadContext::FileHandlerReadContext(bool debug)
	: debug(debug) {}

FileHandlerReadContext::~FileHandlerReadContext() {}

bool FileHandlerReadContext::getDebug() {
	return debug;
}


EchoesFileHandlerReadContext::EchoesFileHandlerReadContext(bool debug,
	int startIndex, int endIndex)
	: FileHandlerReadContext::FileHandlerReadContext(debug),
		startIndex(startIndex), endIndex(endIndex) {}

EchoesFileHandlerReadContext::~EchoesFileHandlerReadContext() {}

int EchoesFileHandlerReadContext::getStartIndex() {
	return startIndex;
}

int EchoesFileHandlerReadContext::getEndIndex() {
	return endIndex;
}


TextFileHandlerReadContext::TextFileHandlerReadContext(
	bool debug, int expectedEntryCount)
	: FileHandlerReadContext::FileHandlerReadContext(debug),
		expectedEntryCount(expectedEntryCount) {}

TextFileHandlerReadContext::~TextFileHandlerReadContext() {}

int TextFileHandlerReadContext::getExpectedEntryCount() {
	return expectedEntryCount;
}


FileHandlerWriteContext::FileHandlerWriteContext(
	bool debug, int startIndex, int endIndex, const string& metaDataSource)
	: debug(debug), startIndex(startIndex), endIndex(endIndex),
		metaDataSource(metaDataSource) {}

FileHandlerWriteContext::~FileHandlerWriteContext() {}

bool FileHandlerWriteContext::getDebug() {
	return debug;
}

int FileHandlerWriteContext::getStartIndex() {
	return startIndex;
}

int FileHandlerWriteContext::getEndIndex() {
	return endIndex;
}

const string& FileHandlerWriteContext::getMetaDataSource() {
	return metaDataSource;
}