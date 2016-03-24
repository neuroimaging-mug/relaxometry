/* 
 * File:   NiftiFileHandler.cpp
 * Author: christiantinauer
 * 
 * Created on August 27, 2014, 1:20 PM
 */

#include "NiftiFileHandler.hpp"

#include <fslio.h>

#include <stdexcept>
#include "../../includes/StringExtensions.h"

NiftiFileHandler::NiftiFileHandler() {
}

NiftiFileHandler::~NiftiFileHandler() {
}

ProcessorData* NiftiFileHandler::Read(const string& filename, FileHandlerReadContext* context) {
	EchoesFileHandlerReadContext* echoesContext = dynamic_cast<EchoesFileHandlerReadContext*>(context);
	if(echoesContext == NULL)
		throw runtime_error("NiftiFileHandler: ReadContext is not of type Echoes.");
	
	FSLIO* source = FslOpen((char*)filename.c_str(), "rb");
	if(source == NULL)
		throw runtime_error("NiftiFileHandler: Could not open file '" + filename + "'.");
	
	nifti_image* nifti_image = source->niftiptr;

	if(context->getDebug())
		nifti_image_infodump(nifti_image);

	int columnCount = nifti_image->nx;
	int rowCount = nifti_image->ny;
	int sliceCount = nifti_image->nz;
	int timeSeriesCount = nifti_image->nt;
	int dataType = nifti_image->datatype;
	
	int startIndex = echoesContext->getStartIndex();
	int endIndex = echoesContext->getEndIndex();
	int count;
	if(startIndex > 0 && endIndex > 0)
		count = endIndex - startIndex + 1;
	else {
		count = timeSeriesCount;
		startIndex = 1;
		endIndex = count;
	}
	
	float echospacing = nifti_image->dt;
	if(nifti_image->time_units == NIFTI_UNITS_USEC)
		echospacing /= 1000;
	else if(nifti_image->time_units == NIFTI_UNITS_SEC)
		echospacing *= 1000;
	
	short t;
	size_t bitsPerVoxel = FslGetDataType(source, &t);
		
	int sliceSize = columnCount * rowCount * bitsPerVoxel / 8;
	int selectedDataPerSlice = sliceSize * count;
	void* selectedData = malloc(sliceCount * selectedDataPerSlice);
	
	FslSeekVolume(source, startIndex - 1);
	FslReadVolumes(source, selectedData, count);
	FslClose(source);
	
	return new ProcessorData(
		startIndex, endIndex,
		columnCount, rowCount, sliceCount, echospacing,
		selectedData, (ProcessorDataDataType)dataType);
}

void NiftiFileHandler::Write(const string& filename, ProcessorData* data, FileHandlerWriteContext* context) {
	FSLIO* source = FslOpen(context->getMetaDataSource().c_str(), "rb");
	FSLIO* destination = FslOpen(filename.c_str(), "wb");
	
	FslCloneHeader(destination, source);
	FslClose(source);
	
	FslSetDim(
		destination, 
		data->getColumnCount(), data->getRowCount(),
		data->getSliceCount(), 1);
	
	FslSetDataType(destination, (short)data->getDataType());
	
	FslWriteAllVolumes(destination, data->getData());
	FslClose(destination);
}