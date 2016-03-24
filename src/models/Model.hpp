#ifndef MODEL_CPP
#define MODEL_CPP

#include "../io/data/ProcessorData.hpp"

#include <string>

using namespace std;

//input, output, length, parameters, parametersLength, constants, constantsLength
//typedef void (*ModelFunction)(float*, float*, int, float*, int, float*, int);

//weights, length, index
//typedef double (*WeightingFunction)(double*, int, int);

class Model {
	private:
		short startModelFunctionID;
		short modelFunctionID;
		short residualWeightingFunctionID;
		short alphaWeightingFunctionID;
		int parametersCount;
		bool needsFlipAnglesMap;
		bool needsT1Map;
		double* echotimes;
		double timeCorrectionFactor;
		double* weights;
		ProcessorData* echoesData;
		ProcessorData* flipAnglesData;
		ProcessorData* t1Data;
		ProcessorDataDataType outputDataType;
		double* parameterStartValues;
		double* parameterBoundaries;
		
	public:
		Model(short startModelFunctionID, short modelFunctionID,
					short residualWeightingFunctionID, short alphaWeightingFunctionID,
					int parametersCount, double timeCorrectionFactor,
					bool needsFlipAnglesMap, bool needsT1Map,
					ProcessorDataDataType outputDataType)
			: startModelFunctionID(startModelFunctionID),
				modelFunctionID(modelFunctionID),
				residualWeightingFunctionID(residualWeightingFunctionID),
				alphaWeightingFunctionID(alphaWeightingFunctionID),
				parametersCount(parametersCount), 
				timeCorrectionFactor(timeCorrectionFactor),
				needsFlipAnglesMap(needsFlipAnglesMap),
				needsT1Map(needsT1Map), outputDataType(outputDataType) {}

		inline short getStartModelFunctionID() {
			return startModelFunctionID;
		}
					
		inline short getModelFunctionID() {
			return modelFunctionID;
		}
			
		inline short getResidualWeightingFunctionID() {
			return residualWeightingFunctionID;
		}
				
		inline short getAlphaWeightingFunctionID() {
			return alphaWeightingFunctionID;
		}
				
		inline int getParametersCount() {
			return parametersCount;
		}
		
		inline double getTimeCorrectionFactor() {
			return timeCorrectionFactor;
		}
		
		inline bool getNeedsFlipAnglesMap() { 
			return needsFlipAnglesMap;
		}
		
		inline bool getNeedsT1Map() {
			return needsT1Map;
		}

		inline double* getEchotimes() {
			return echotimes;
		}

		inline double* getWeights() {
			return weights;
		}
		
		inline ProcessorData* getEchoesData() {
			return echoesData;
		}
		
		inline ProcessorData* getFlipAnglesData() {
			return flipAnglesData;
		}
		
		inline ProcessorData* getT1Data() {
			return t1Data;
		}
		
		inline ProcessorDataDataType getOutputDataType() {
			return outputDataType;
		}
		
		inline double* getParameterStartValues() {
			return parameterStartValues;
		}
		
		inline double* getParameterBoundaries() {
			return parameterBoundaries;
		}
				
		inline void setEchotimes(double* et) {
			echotimes = et;
		} 
		
		inline void setWeights(double* w) {
			weights = w;
		}
		
		inline void setEchoesData(ProcessorData* ed) {
			echoesData = ed; 
		}
		
		inline void setFlipAnglesData(ProcessorData* fad) {
			flipAnglesData = fad; 
		}
		
		inline void setT1Data(ProcessorData* t1d) {
			t1Data = t1d; 
		}
		
		inline void setParameterStartValues(double* startValues) {
			parameterStartValues = startValues;
		}
		
		inline void setParameterBoundaries(double* boundaries) {
			parameterBoundaries = boundaries;
		}
				
		static Model* CreateModelFromArgument(const string& argument,
			bool forCUDAProcessor, bool useWeights);
};

#endif /* MODEL_CPP */