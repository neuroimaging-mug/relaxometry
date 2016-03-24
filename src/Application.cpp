#include "Application.hpp"

#include "cuda/CUDADevicesService.hpp"
#include "io/handler/IFileHandler.hpp"
#include "io/handler/TextFileHandler.hpp"
#include "io/data/ProcessorData.hpp"
#include "models/Model.hpp"
#include "processors/ProcessorContext.hpp"
#include "processors/IProcessor.hpp"

#include <string>
#include <stdio.h>
#include <stdexcept>
#include <vector>
#include <limits>

using namespace std;

int main(int argc, const char** argv) {	
	//handle arguments
	ezOptionParser optionParser;
	setupEzOptionParser(optionParser);
	
	optionParser.parse(argc, argv);

  if(optionParser.isSet("-h")) {
    showEzOptionParserUsage(optionParser);
    return EXIT_FAILURE;
  }

	if(optionParser.isSet("-lcdevs")) {
		vector<string> devices = CUDADevicesService::listCUDADevices();
		for(int i = 0; i < devices.size(); ++i)
			cout << devices[i] << ".\n";
    return EXIT_SUCCESS;
  }
	
	vector<string> badOptions;
	if(!optionParser.gotRequired(badOptions)) {
		for(int i = 0; i < badOptions.size(); ++i)
			cerr << "ERROR: Missing required option " << badOptions[i] << "." << endl;
		
		showEzOptionParserUsage(optionParser);
		return EXIT_FAILURE;
	}

	if(!optionParser.gotExpected(badOptions)) {
		for(int i = 0; i < badOptions.size(); ++i)
			cerr << "ERROR: Got unexpected number of arguments for option " << badOptions[i] << "." << endl;

		showEzOptionParserUsage(optionParser);
		return EXIT_FAILURE;
	}

	if(optionParser.lastArgs.size() < 2) {
		cerr << "ERROR: Expected at least two arguments. Input and output file." << endl;
		showEzOptionParserUsage(optionParser);
		return EXIT_FAILURE;
	}
	
	//setup cuda device
	int deviceCount = CUDADevicesService::listCUDADevices().size();
	if(deviceCount > 0) {
		int device;
		optionParser.get("--dev")->getInt(device);
		try {
			CUDADevicesService::setCUDADevice(device);
		} catch(const runtime_error& ex) {
			cerr << ex.what();
		}
		cout << "CUDA device set: " << CUDADevicesService::getCurrentCUDADevice() << endl;
	}
	
	//handle common arguments
	bool debug = optionParser.isSet("+debug");
	bool useDoublePrecision = optionParser.isSet("+usedoubleprecision");
  const string echoesFilename(optionParser.lastArgs[0]->c_str());
	const string parameter1Filename(optionParser.lastArgs[1]->c_str());
	
	int startIndex = -1;
  optionParser.get("--startIndex")->getInt(startIndex);
  int endIndex = -1;
  optionParser.get("--endIndex")->getInt(endIndex);
	float threshold = -numeric_limits<float>::max();
	if(optionParser.isSet("--threshold"))
		optionParser.get("--threshold")->getFloat(threshold);
	int profileLength = 0;
	optionParser.get("--profilelength")->getInt(profileLength);
	int threadCount = 0;
	optionParser.get("--threadcount")->getInt(threadCount);
	float minGoF = 0.f;
	optionParser.get("--mingof")->getFloat(minGoF);
	float stepSizeTolerance = 0.f;
	optionParser.get("--stepsizetolerance")->getFloat(stepSizeTolerance);
	float errorReductionTolerance = 0.f;
	optionParser.get("--errorreductiontolerance")->getFloat(errorReductionTolerance);
	
	//processor argument
	string processorArgument;
	optionParser.get("-p")->getString(processorArgument);
	
	//model argument
	string modelArgument;
	optionParser.get("-m")->getString(modelArgument);
	
	//echotimes
	string echotimesFilename("");
	if(optionParser.isSet("--echotimes"))
		optionParser.get("--echotimes")->getString(echotimesFilename);
	
	//weights
	bool useWeights = optionParser.isSet("--weights");
	string weightsFilename("");
	if(useWeights)
		optionParser.get("--weights")->getString(weightsFilename);
	
	//startvalues
	vector<double> startValues;
	optionParser.get("--startvalues")->getDoubles(startValues);
	
	//boundaries
	vector<double> boundaries;
	optionParser.get("--boundaries")->getDoubles(boundaries);
	
	try {		
		//create processor
		IProcessor* processor = IProcessor::CreateProcessorFromArgument(processorArgument);
		if(processor->IsCUDAProcessor() && deviceCount == 0) {
			delete processor;
			cerr << "ERROR: No CUDA device found. CUDA processors cannot be used." << endl;
			return EXIT_FAILURE;
		}
		
		//create model
		Model* model = Model::CreateModelFromArgument(modelArgument, processor->IsCUDAProcessor(), useWeights);
		int parametersCount = model->getParametersCount();
		double timeCorrectionFactor = model->getTimeCorrectionFactor();
		
		//output files
		int lastArgsSize = optionParser.lastArgs.size();
		
		bool outputParameter2 = lastArgsSize >= 3;
		const string parameter2Filename(outputParameter2 ? optionParser.lastArgs[2]->c_str() : "");
	
		bool outputParameter3 = parametersCount == 3 && lastArgsSize >= 4;
		const string parameter3Filename(outputParameter3 ? optionParser.lastArgs[3]->c_str() : "");
		
		bool outputGoodnessOfFit = 
			parametersCount == 2 && lastArgsSize >= 4 || parametersCount == 3 && lastArgsSize >= 5;
		const string goodnessOfFitFilename(
			outputGoodnessOfFit 
				? optionParser.lastArgs[parametersCount == 2 ? 3 : 4]->c_str() 
				: "");
		
		//set startvalues
		double startValuesArray[parametersCount];
		for(int index = 0; index < parametersCount; ++index)
			startValuesArray[index] = startValues[index];
		model->setParameterStartValues(startValuesArray);
		
		//set boundaries
		double boundariesArray[parametersCount * 2];
		for(int index = 0; index < parametersCount * 2; ++index)
			boundariesArray[index] = boundaries[index];
		model->setParameterBoundaries(boundariesArray);
		
		//read echoes data
		IFileHandler* echoesHandler = IFileHandler::CreateFileHandlerFromFilenameExtension(echoesFilename);
		EchoesFileHandlerReadContext* echoesHandlerReadContext = new EchoesFileHandlerReadContext(debug, startIndex, endIndex);
		ProcessorData* echoesData = echoesHandler->Read(echoesFilename, echoesHandlerReadContext);
		model->setEchoesData(echoesData);
		delete echoesHandlerReadContext;
		startIndex = echoesData->getStartIndex();
		endIndex = echoesData->getEndIndex();
		
		//read flipangles
		ProcessorData* flipAnglesData = NULL;
		if(model->getNeedsFlipAnglesMap()) {
			if(!optionParser.isSet("--flipanglesmap")) {
				cerr << "ERROR: Missing required option --flipanglesmap." << endl;
				return EXIT_FAILURE;
			}
			
			string flipAnglesMapFilename;
			optionParser.get("--flipanglesmap")->getString(flipAnglesMapFilename);
			
			IFileHandler* flipAnglesHandler =
				IFileHandler::CreateFileHandlerFromFilenameExtension(flipAnglesMapFilename);
			FileHandlerReadContext* flipAnglesReadContext = (FileHandlerReadContext*)
				dynamic_cast<TextFileHandler*>(flipAnglesHandler) != NULL 
					? (FileHandlerReadContext*)new TextFileHandlerReadContext(
							debug, echoesData->getColumnCount() * echoesData->getRowCount())
					:	(FileHandlerReadContext*)new EchoesFileHandlerReadContext(debug, 1, 1);
			flipAnglesData = flipAnglesHandler->Read(flipAnglesMapFilename, flipAnglesReadContext);
			model->setFlipAnglesData(flipAnglesData);
			delete flipAnglesReadContext;
			delete flipAnglesHandler;
		}
		
		//read t1
		ProcessorData* t1Data = NULL;
		if(model->getNeedsT1Map()) {
			if(!optionParser.isSet("--t1map")) {
				cerr << "ERROR: Missing required option --t1map." << endl;
				showEzOptionParserUsage(optionParser);
				return EXIT_FAILURE;
			}
			
			string t1MapFilename;
			optionParser.get("--t1map")->getString(t1MapFilename);
			
			IFileHandler* t1Handler = IFileHandler::CreateFileHandlerFromFilenameExtension(t1MapFilename);
			FileHandlerReadContext* t1ReadContext =
				dynamic_cast<TextFileHandler*>(t1Handler) != NULL 
					? (FileHandlerReadContext*)new TextFileHandlerReadContext(
							debug, echoesData->getColumnCount() * echoesData->getRowCount())
					:	(FileHandlerReadContext*)new EchoesFileHandlerReadContext(debug, 1, 1);
			t1Data = t1Handler->Read(t1MapFilename, t1ReadContext);
			model->setT1Data(t1Data);
			delete t1ReadContext;
			delete t1Handler;
		}
				
		//read echotimes if given
		int count = endIndex - startIndex + 1;
		double* echotimes = NULL;
		if(echotimesFilename.compare("") != 0) {
			IFileHandler* echotimesHandler = IFileHandler::CreateFileHandlerFromFilenameExtension(echotimesFilename);
			TextFileHandlerReadContext* echotimesReadContext = new TextFileHandlerReadContext(debug, count);
			ProcessorData* echotimesData = echotimesHandler->Read(echotimesFilename, echotimesReadContext);
			echotimes = (double*)echotimesData->getData();
			delete echotimesReadContext;
			delete echotimesHandler;
			delete echotimesData;
		}
		if(echotimes == NULL) {
			echotimes = new double[count];
			for(int index = 0; index < count; ++index)
				echotimes[index] = echoesData->getEchospacing() * (startIndex + index) * timeCorrectionFactor;
		} else
			for(int index = 0; index < count; ++index)
				echotimes[index] = echotimes[index] * timeCorrectionFactor;
		model->setEchotimes(echotimes);
		
		//read weights if given
		int sliceCount = echoesData->getSliceCount();
		int weightsCount = sliceCount * count;
		double* weights = NULL;
		if(weightsFilename.compare("") != 0) {
			IFileHandler* weightsHandler = IFileHandler::CreateFileHandlerFromFilenameExtension(weightsFilename);
			TextFileHandlerReadContext* weightsReadContext = new TextFileHandlerReadContext(
				debug, weightsCount);
			ProcessorData* weightsData = weightsHandler->Read(weightsFilename, weightsReadContext);
			weights = (double*)weightsData->getData();
			delete weightsReadContext;
			delete weightsHandler;
			delete weightsData;
		}
		if(weights == NULL) {
			weights = new double[weightsCount];
			for(int index = 0; index < weightsCount; ++index)
				weights[index] = 1.;
		}
		model->setWeights(weights);
		
		//execute processor
		ProcessorContext* processorContext = new ProcessorContext(
			debug, useDoublePrecision, 
			outputParameter2, outputParameter3, outputGoodnessOfFit,
			threshold, profileLength, threadCount,
			minGoF, stepSizeTolerance, errorReductionTolerance);
		vector<ProcessorData*> outputData = processor->Execute(model, processorContext);

		//write data
		FileHandlerWriteContext* writeContext = 
			new FileHandlerWriteContext(debug, startIndex, endIndex, echoesFilename);
		echoesHandler->Write(parameter1Filename, outputData[0], writeContext);
		if(outputParameter2)
			echoesHandler->Write(parameter2Filename, outputData[1], writeContext);
		if(outputParameter3)
			echoesHandler->Write(parameter3Filename, outputData[2], writeContext);
		if(outputGoodnessOfFit)
			echoesHandler->Write(goodnessOfFitFilename, outputData[outputParameter3 ? 3 : 2], writeContext);
		
		//cleanup
		free(outputData[0]->getData());
		for(int index = 0; index < outputData.size(); ++index)
			delete outputData[index];
		delete echoesHandler;
		if(flipAnglesData != NULL)
			delete flipAnglesData;
		if(t1Data != NULL)
			delete t1Data;
		delete[] echotimes;
		delete[] weights;
		delete model;
		delete processorContext;
		delete processor;
	} catch(const runtime_error& ex) {
		cerr << "ERROR: " << ex.what() << endl;
		return EXIT_FAILURE;
	}
	
	return EXIT_SUCCESS;
}

void setupEzOptionParser(ezOptionParser& optionParser) {
	//base
	optionParser.overview = "Generates T1/T2/T2* maps using Levenberg-Marquardt or Levenberg-Marquardt-Fletcher fitting for the Exponential (T1/T2/T2star) or Lukzen-Savelov (T2) model using C or CUDA.";
  optionParser.syntax = "relaxometry -p clr|cudalr|clm|cudalm|clmf|cudalmf -m lrr2|lrrt2|exp3pr1|exp3pt1|exp2pr1|exp2pt1|expr2|expt2 [OPTIONS] INPUT_FILE OUTPUT_FILE(S)";
  optionParser.example = "relaxometry -t 100 -p cudalmf -m ls --threadcount 64 --flipanglesmap ../data/nii/input/T2/1/FAmap.txt --t1map ../data/nii/input/T2/1/T1map.txt ../data/nii/input/T2/1/mse_brain_01_3.nii.gz ../data/nii/output/T2/1/T2_cudalmf_ls_1.nii.gz ../data/nii/output/T2/1/M0_cudalmf_ls_1.nii.gz ../data/nii/output/T2/1/GoF_cudalmf_ls_1.nii.gz\n\n";
  optionParser.footer = "Christian Tinauer (christian.tinauer@student.tugraz.at).\nBased on the work of Christof Sirk.\nThis program is free and without warranty.\n";
	
	//common
	optionParser.add(
		"", NOT_REQUIRED_OPTION, 0, 0, "Display usage instructions.", 
		"-h", "-help",  "--help", "--usage");
	optionParser.add(
		"", NOT_REQUIRED_OPTION, 0, 0, "Print debug output.",
		"+d", "+dbg", "+debug");
	optionParser.add(
		"", NOT_REQUIRED_OPTION, 0, 0, "Use double precision for calculations (only clm, clmf).",
		"+usedoubleprecision");
	optionParser.add(
		"", NOT_REQUIRED_OPTION, 0, 0, "List CUDA devices.",
		"-lcdevs", "--listcudadevices");
	optionParser.add(
		"0", NOT_REQUIRED_OPTION, 1, 0, "CUDA device to be used.",
		"--dev");
	optionParser.add(
		"64", NOT_REQUIRED_OPTION, 1, 0, "Thread count for CUDA kernel calls.",
		"-tc", "--threadcount");
		
	//processor
	optionParser.add(
		"", REQUIRED_OPTION, 1, 0, "Processor: clr, clm, clmf, cudalr, cudalm, cudalmf.",
		"-p", "--processor");
	
	//model
	optionParser.add(
		"", REQUIRED_OPTION, 1, 0, "Model: exp2pr1, exp2pt1, exp3pr1, exp3pt1, expr2, expt2, ls, lrr2, lrt2.",
		"-m", "--model");
	
	//processor infos
	optionParser.add(
		"-1", NOT_REQUIRED_OPTION, 1, 0, "Index of first echo which should be used.",
		"-s", "--startIndex");
  optionParser.add(
		"-1", NOT_REQUIRED_OPTION, 1, 0, "Index of last echo which should be used.",
    "-e", "--endIndex");
	optionParser.add(
		"1", NOT_REQUIRED_OPTION, 1, 0, "Profile length for generating function (Lukzen-Savelov-Model). Values from 1 to 13.",
    "-pl", "--profilelength");
	optionParser.add(
		"0", NOT_REQUIRED_OPTION, 1, 0, "Threshold defines valid pixel of first echo.",
    "-t", "--threshold");
	optionParser.add(
		"", NOT_REQUIRED_OPTION, 1, 0, "Echotimes.",
    "-et", "--echotimes");
	optionParser.add(
		"", NOT_REQUIRED_OPTION, 1, 0, "Normalized weights for fit.",
    "-w", "--weights");
	optionParser.add(
		"", NOT_REQUIRED_OPTION, 1, 0, "T1 map path.",
		"--t1map", "--t1");
  optionParser.add(
		"", NOT_REQUIRED_OPTION, 1, 0, "Flip angles map path.",
		"--flipanglesmap", "--flipangles");
	optionParser.add(
		"1000,800,1.98", NOT_REQUIRED_OPTION, -1, ',', "Start values for T1, M0, B1 fit.",
		"--startvalues");
	optionParser.add(
		"-30000,30000,0,15000,1.5,2", NOT_REQUIRED_OPTION, -1, ',', "Min/max values for T1/T2/T2star, M0, B1.",
		"--boundaries");
	optionParser.add(
		"0.8", NOT_REQUIRED_OPTION, 1, 0, "Minimum value for Goodness-of-Fit. If not reached fit is restarted with different starting values (T1 only).",
    "--mingof");
	
	//stopping criterions
	optionParser.add(
		"1e-7", NOT_REQUIRED_OPTION, 1, 0, "Stopping tolerance for update step. All parts of step have to be smaller than given value.",
    "--stepsizetolerance");
	optionParser.add(
		"1e-7", NOT_REQUIRED_OPTION, 1, 0, "Stopping tolerance for error reduction.",
    "--errorreductiontolerance");
}

void showEzOptionParserUsage(ezOptionParser& optionParser) {
	string usage;
	optionParser.getUsage(usage);
	cout << usage;
}