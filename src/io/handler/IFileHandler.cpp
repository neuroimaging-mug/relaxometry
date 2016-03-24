#include "IFileHandler.hpp"

#include "DicomFileHandler.hpp"
#include "TextFileHandler.hpp"
#include "NiftiFileHandler.hpp"

IFileHandler* IFileHandler::CreateFileHandlerFromFilenameExtension(const string& filename) {
	if(IFileHandler::HasSuffix(filename, ".DCM") || IFileHandler::HasSuffix(filename, ".dcm"))
		return new DicomFileHandler();
	else if(IFileHandler::HasSuffix(filename, ".TXT") || IFileHandler::HasSuffix(filename, ".txt"))
		return new TextFileHandler();
	else if(IFileHandler::HasSuffix(filename, ".NII") || IFileHandler::HasSuffix(filename, ".nii") ||
					IFileHandler::HasSuffix(filename, ".NII.GZ") || IFileHandler::HasSuffix(filename, ".nii.gz"))
		return new NiftiFileHandler();
	else
		throw runtime_error("Not supported file type: " + filename);
}

bool IFileHandler::HasSuffix(const string& text, const string& suffix) {
	return	text.size() >= suffix.size() &&
					text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}