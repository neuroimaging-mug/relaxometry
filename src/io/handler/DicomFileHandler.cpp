/* 
 * File:   DicomFileHandler.cpp
 * Author: christiantinauer
 * 
 * Created on August 27, 2014, 1:15 PM
 */

#include "DicomFileHandler.hpp"

#include <gdcm-2.6/gdcmAttribute.h>
#include <gdcm-2.6/gdcmAnonymizer.h>
#include <gdcm-2.6/gdcmImageWriter.h>

#include <stdexcept>

DicomFileHandler::DicomFileHandler() {
}

DicomFileHandler::~DicomFileHandler() {
}

ProcessorData* DicomFileHandler::Read(const string& filename, FileHandlerReadContext* context) {
	EchoesFileHandlerReadContext* echoesContext = dynamic_cast<EchoesFileHandlerReadContext*>(context);
	if(echoesContext == NULL)
		throw runtime_error("DicomFileHandler: ReadContext is not of type Echoes.");
	
	int startIndex = echoesContext->getStartIndex();
	int endIndex = echoesContext->getEndIndex();

	int count = endIndex - startIndex + 1;
	int index = startIndex;

	char concreteFilename[512];
	unsigned short* data;
	char* buffer;
	long imageSize = 0;

	int columnCount;
	int rowCount;
	float echospacing;
	
	for(int i = 0; i < count; ++i) {
		sprintf(concreteFilename, filename.c_str(), index);
		ImageReader imageReader;
		imageReader.SetFileName(concreteFilename);
		if(!imageReader.Read())
			throw runtime_error("DicomFileHandler: Could not read file '" + string(concreteFilename) + "'.");

		const Image &image = imageReader.GetImage();
		if(i == 0) {
			imageSize = image.GetBufferLength();
			buffer = (char*)malloc(imageSize);
			data = (unsigned short*)malloc(imageSize * count);
			
			File& file = imageReader.GetFile();
			DataSet& ds = file.GetDataSet();
			columnCount = GetAttributeValue<0x0028, 0x0011>(ds);
			rowCount = GetAttributeValue<0x0028, 0x0010>(ds);
			echospacing = GetAttributeValue<0x0018, 0x0081>(ds);
		}
		
		if(!image.GetBuffer(buffer))
			throw runtime_error("DicomFileHandler: Reading buffer failed.");
			
		int offset = i * imageSize / sizeof(unsigned short);
		memcpy(data + offset, buffer, imageSize);

		++index;
	}
	
	free(buffer);
	
	return new ProcessorData(
		startIndex, endIndex, columnCount, rowCount, 1,
		echospacing, data, UINT16);
}

void DicomFileHandler::Write(const string& filename, ProcessorData* data, FileHandlerWriteContext* context) {
	char metaDataSource[512];
	sprintf(metaDataSource, context->getMetaDataSource().c_str(), context->getStartIndex());
	
	ImageReader imageReader;
  imageReader.SetFileName(metaDataSource);
  if(!imageReader.Read())
		throw runtime_error("DicomFileHandler: Could not read file '" + string(metaDataSource) + "'.");
 
	//TODO: Implement handling for different data types.
	unsigned short* byteValue;
	if(data->getDataType() == UINT16)
		byteValue = (unsigned short*)data->getData();
	else {
		int size = data->getColumnCount() * data->getRowCount();
		byteValue = new unsigned short[size];
		float* floatData = (float*)data->getData();
		for(int index = 0; index < size; ++index)
			byteValue[index] = round(floatData[index]);
	}
	
  Image &image = imageReader.GetImage();
	DataElement pixeldata(Tag(0x7fe0, 0x0010));
	pixeldata.SetByteValue((char*)byteValue, image.GetBufferLength());
  image.SetDataElement(pixeldata);

  ImageWriter w;
  w.SetFile(imageReader.GetFile());
	w.SetImage(image);
	w.SetFileName(filename.c_str());
  
  if(!w.Write())
		throw runtime_error("DicomFileHandler: Could not write file '" + filename + "'.");
}

void DicomFileHandler::Anonymize(const string& filename) {
  Reader reader;
  reader.SetFileName(filename.c_str());
  if(!reader.Read())
		throw runtime_error("DicomFileHandler: Could not read file '" + filename + "'.");
  
	Anonymizer anonymizer;
  anonymizer.SetFile(reader.GetFile());

  anonymizer.Replace(Tag(0x0008, 0x0012), "19000101");	// Instance Creation Date
  anonymizer.Replace(Tag(0x0008, 0x0013), "00000000");	// Instance Creation Time
  anonymizer.Replace(Tag(0x0008, 0x0020), "19000101");	// Study Date
  anonymizer.Replace(Tag(0x0008, 0x0021), "19000101");	// Series Date
  anonymizer.Replace(Tag(0x0008, 0x0022), "19000101");	// Acquisition Date
  anonymizer.Replace(Tag(0x0008, 0x0023), "19000101");	// Content Date
  anonymizer.Replace(Tag(0x0008, 0x0030), "00000000");	// Study Time
  anonymizer.Replace(Tag(0x0008, 0x0031), "00000000");	// Series Time
  anonymizer.Replace(Tag(0x0008, 0x0032), "00000000");	// Acquisition Time
  anonymizer.Replace(Tag(0x0008, 0x0033), "00000000");	// Content Time
  anonymizer.Replace(Tag(0x0008, 0x0080), "Anonymous");	// Institution Name
  anonymizer.Replace(Tag(0x0008, 0x0081), "Anonymous"); // Institution Address
  anonymizer.Replace(Tag(0x0008, 0x1010), "Anonymous"); // Station Name
  anonymizer.Replace(Tag(0x0008, 0x1030), "Anonymous"); // Study Description
  anonymizer.Replace(Tag(0x0008, 0x1050), "Anonymous"); // Performing Physician's Name
  anonymizer.Replace(Tag(0x0010, 0x0010), "Anonymous"); // Patient's Name
  anonymizer.Replace(Tag(0x0010, 0x0020), "Anonymous"); // Patient ID
  anonymizer.Replace(Tag(0x0010, 0x0030), "Anonymous"); // Patient's Birth Date
  anonymizer.Replace(Tag(0x0010, 0x0040), "U");					// Patient's Sex
  anonymizer.Replace(Tag(0x0010, 0x1010), "0");					// Patient's Age
  anonymizer.Replace(Tag(0x0010, 0x1030), "0");					// Patient's Weight
  anonymizer.Replace(Tag(0x0032, 0x1060), "Anonymous");	// Requested Procedure Description
  anonymizer.Replace(Tag(0x0040, 0x0254), "Anonymous");	// Performed Procedure Step Description

  Writer writer;
  writer.SetFileName(filename.c_str());
	writer.SetFile(reader.GetFile());
  if(!writer.Write())
    throw runtime_error("DicomFileHandler: Could not write file '" + filename + "'.");
}

template<unsigned short Group, unsigned short Element> int DicomFileHandler::GetAttributeValue(DataSet& ds)
{
  Attribute<Group, Element> attribute;
  attribute.SetFromDataSet(ds);
  return attribute.GetValue();
}