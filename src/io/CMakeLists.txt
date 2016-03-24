project(io)

find_package(GDCM REQUIRED)
include(${GDCM_USE_FILE})

find_package(NIFTI REQUIRED)
include(${NIFTI_USE_FILE})

set(SOURCE_FILES 
	data/ProcessorData
	handler/FileHandlerContext
	handler/IFileHandler
	handler/DicomFileHandler
	handler/NiftiFileHandler
	handler/TextFileHandler
)

add_library(relaxometryio_static STATIC ${SOURCE_FILES})
target_link_libraries(relaxometryio_static ${GDCM_LIB} ${NIFTI_LIB})