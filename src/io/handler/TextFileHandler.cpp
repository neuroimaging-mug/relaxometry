/* 
 * File:   MatlabHandler.cpp
 * Author: christiantinauer
 * 
 * Created on August 29, 2014, 4:28 PM
 */

#include "TextFileHandler.hpp"

#include "../../includes/StringExtensions.h"
#include <stdexcept>
#include <fstream>
#include <cstdlib>

TextFileHandler::TextFileHandler() {
}

TextFileHandler::~TextFileHandler() {
}

ProcessorData* TextFileHandler::Read(const string& filename, FileHandlerReadContext* context) {
	TextFileHandlerReadContext* textContext = dynamic_cast<TextFileHandlerReadContext*>(context);
	if(textContext == NULL)
		throw runtime_error("TextFileHandler: ReadContext is not of type Text.");
	
	int entryCount = textContext->getExpectedEntryCount();
	double* data = new double[entryCount];
	
	ifstream infile(filename.c_str());

	int index = 0;
	while(infile) {
		string line;
		if(!getline(infile, line))
			break;

		istringstream ss(line);

		while(ss) {
			string s;
			if(!getline(ss, s, ' '))
				break;

			if(s.length() == 0)
				continue;

			data[index++] = atof(s.c_str());
		}
	}
	
	if(index != entryCount)
		throw runtime_error("TextFileHandler: Failed to read '" + filename + "'. " +
			"Expected length: " + to_string(entryCount) + ". "
			"Actual length: " + to_string(index) + ".");
	
	return new ProcessorData(-1, -1, -1, -1, -1, -1.f, data, FLOAT64);
}

void TextFileHandler::Write(const string& filename, ProcessorData* data, FileHandlerWriteContext* context) {
	throw runtime_error("TextFileHandler::Write not implemented.");
}