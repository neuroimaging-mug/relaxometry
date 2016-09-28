project(app)

find_package(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	#-G #device-debug, problems with pointers global/shared memory -> works since cc 2.0
	#-g #host-debug
	#-gencode=arch=compute_12,code=sm_12
	-gencode=arch=compute_20,code=sm_20
	-gencode=arch=compute_50,code=sm_50
)

IF(NO_CUDA)
	add_definitions(-DNO_CUDA)
ENDIF(NO_CUDA)

#preprocessor statement for switching between global and shared memory for calculations (CUDA)
#add_definitions(-DUSE_SHARED_MEMORY)

IF(NO_CUDA)
set(LIB_SOURCE_FILES
	includes/StringExtensions
	io/data/ProcessorData
	#Models
	models/Model
	models/c/ModelService
	models/c/Model
	#Processor base
	processors/ProcessorContext
	processors/IProcessor	
	processors/ProcessorService
	#C Processors
	processors/c/LinearRegression/CLinearRegression	
	processors/c/LinearRegression/CLinearRegressionProcessor
	processors/c/LevenbergMarquardt/CJacobian
	processors/c/LevenbergMarquardt/CCalculateStepVariables
	processors/c/LevenbergMarquardt/CLevenbergMarquardtCore
	processors/c/LevenbergMarquardt/CLevenbergMarquardt
	processors/c/LevenbergMarquardt/CLevenbergMarquardtProcessor
	processors/c/LevenbergMarquardtFletcher/CLevenbergMarquardtFletcherCore
	processors/c/LevenbergMarquardtFletcher/CLevenbergMarquardtFletcher
	processors/c/LevenbergMarquardtFletcher/CLevenbergMarquardtFletcherProcessor
)

	cuda_add_library(relaxometry_ccore SHARED ${LIB_SOURCE_FILES})
	cuda_add_library(relaxometry_ccore_static STATIC ${LIB_SOURCE_FILES})
ELSE(NO_CUDA)
set(LIB_SOURCE_FILES
	includes/StringExtensions
	cuda/CUDADevicesService
	io/data/ProcessorData
	#Models
	models/Model
	models/c/ModelService
	models/c/Model
	models/cuda/Complex.cuh
	models/cuda/Statics.cuh
	models/cuda/Exponential2ParametersR1.cuh
	models/cuda/Exponential2ParametersT1.cuh
	models/cuda/Exponential3ParametersR1.cuh
	models/cuda/Exponential3ParametersT1.cuh
	models/cuda/ExponentialR2.cuh
	models/cuda/ExponentialT2.cuh
	models/cuda/LinearRegressionR2.cuh
	models/cuda/LinearRegressionT2.cuh
	models/cuda/LukzenSavelov.cuh
	models/cuda/SimpleStartValues.cuh
	models/cuda/LinearWeighting.cuh
	models/cuda/InverseMinimumWeighting.cuh
	models/cuda/InverseQuadraticWeighting.cuh
	models/cuda/Model.cuh
	#Processor base
	processors/ProcessorContext
	processors/IProcessor	
	processors/ProcessorService
	#C Processors
	processors/c/LinearRegression/CLinearRegression	
	processors/c/LinearRegression/CLinearRegressionProcessor
	processors/c/LevenbergMarquardt/CJacobian
	processors/c/LevenbergMarquardt/CCalculateStepVariables
	processors/c/LevenbergMarquardt/CLevenbergMarquardtCore
	processors/c/LevenbergMarquardt/CLevenbergMarquardt
	processors/c/LevenbergMarquardt/CLevenbergMarquardtProcessor
	processors/c/LevenbergMarquardtFletcher/CLevenbergMarquardtFletcherCore
	processors/c/LevenbergMarquardtFletcher/CLevenbergMarquardtFletcher
	processors/c/LevenbergMarquardtFletcher/CLevenbergMarquardtFletcherProcessor
	#CUDA Processors
	processors/cuda/ErrorHandling.cuh
	processors/cuda/MemoryHandling.cuh
	processors/cuda/LinearRegression/CudaLinearRegression
	processors/cuda/LinearRegression/CudaLinearRegression.cu
	processors/cuda/LinearRegression/CudaLinearRegressionProcessor
	processors/cuda/LevenbergMarquardtFletcher/CudaCalculateAdjustedRsquareValue.cuh
	processors/cuda/LevenbergMarquardtFletcher/CudaCalculateInverseMatrix.cuh
	processors/cuda/LevenbergMarquardtFletcher/CudaJacobian.cu
	processors/cuda/LevenbergMarquardtFletcher/CudaStepVariables.cuh
	processors/cuda/LevenbergMarquardtFletcher/CudaLevenbergMarquardtFletcherCore.cu
	processors/cuda/LevenbergMarquardtFletcher/CudaLevenbergMarquardtFletcher	
	processors/cuda/LevenbergMarquardtFletcher/CudaLevenbergMarquardtFletcher.cu
	processors/cuda/LevenbergMarquardtFletcher/CudaLevenbergMarquardtFletcherProcessor
	processors/cuda/LevenbergMarquardt/CudaLevenbergMarquardtCore.cu
	processors/cuda/LevenbergMarquardt/CudaLevenbergMarquardt	
	processors/cuda/LevenbergMarquardt/CudaLevenbergMarquardt.cu
	processors/cuda/LevenbergMarquardt/CudaLevenbergMarquardtProcessor
)

	cuda_add_library(relaxometrycore SHARED ${LIB_SOURCE_FILES})
	cuda_add_library(relaxometrycore_static STATIC ${LIB_SOURCE_FILES})
ENDIF(NO_CUDA)

set(APP_SOURCE_FILES
	includes/ezOptionParser
	Application
)

add_subdirectory(io)

IF(NO_CUDA)
	cuda_add_executable(relaxometry_c ${APP_SOURCE_FILES})
	target_link_libraries(relaxometry_c relaxometry_ccore_static relaxometryio_static ${GDCM_LIB} ${NIFTI_LIB})
ELSE(NO_CUDA)
	cuda_add_executable(relaxometry ${APP_SOURCE_FILES})
	target_link_libraries(relaxometry relaxometrycore_static relaxometryio_static ${GDCM_LIB} ${NIFTI_LIB})
ENDIF(NO_CUDA)