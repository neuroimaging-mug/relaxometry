/* 
 * File:   clmf_expr2.cpp
 * Author: christiantinauer
 *
 * Created on September 23, 2015, 1:54 PM
 */

#include "mex.h"

#include "../processors/IProcessor.hpp"
#include "../processors/ProcessorContext.hpp"
#include "../models/Model.hpp"
#include "../io/data/ProcessorData.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]) {
	//create processor
	IProcessor* processor = IProcessor::CreateProcessorFromArgument("cudalmf");
		
	//create model
	Model* model = Model::CreateModelFromArgument("expr2", processor->IsCUDAProcessor(), true);
	
	//startvalues
	double startValues[2] = {};
	model->setParameterStartValues(startValues);
	
	//boundaries
	double boundaries[4] = {0, 15000, 0, 15000};
	model->setParameterBoundaries(boundaries);
		
	//echotimes
	int echotimesLength = mxGetM(prhs[0]);
	double* echotimes = (double*)mxGetData(prhs[0]);
	model->setEchotimes(echotimes);
	
	//echoesdata
	int echoesDataLength = mxGetM(prhs[1]);
	int sliceSize = echoesDataLength / echotimesLength;
	float* echoesData= (float*)mxGetData(prhs[1]);	
	ProcessorData* echoesProcessorData = new ProcessorData(
		1, echotimesLength, 1, sliceSize, 
		1, 0.f, echoesData, FLOAT32);
	model->setEchoesData(echoesProcessorData);
	
	//weights
	double* weights = (double*)mxGetData(prhs[2]);
	model->setWeights(weights);
	
	//threshold
	double threshold = mxGetScalar(prhs[3]);
	
	//processor context
	ProcessorContext* context = new ProcessorContext(
      false, true, true, false, true, threshold, 1, 64, 0.8, 1e-7, 1e-7);
	
	//execute
	vector<ProcessorData*> result = processor->Execute(model, context);
	
	plhs[0] = mxCreateDoubleMatrix(sliceSize, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(sliceSize, 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(sliceSize, 1, mxREAL);
	
	float* r2mapRaw = (float*)result[0]->getData();
	double* r2map = mxGetPr(plhs[0]);
	for(int index = 0; index < sliceSize; ++index)
		r2map[index] = (double)r2mapRaw[index];
	
	float* m0mapRaw = (float*)result[1]->getData();
	double* m0map = mxGetPr(plhs[1]);
	for(int index = 0; index < sliceSize; ++index)
		m0map[index] = (double)m0mapRaw[index];
	
	float* GOFmapRaw = (float*)result[2]->getData();
	double* GOFmap = mxGetPr(plhs[2]);
	for(int index = 0; index < sliceSize; ++index)
		GOFmap[index] = (double)GOFmapRaw[index];
	
	free(result[0]->getData());
	delete result[0];
	delete result[1];
	delete result[2];
	delete context;
	delete model;
	delete processor;
}