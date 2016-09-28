cmake_minimum_required (VERSION 2.8)

project(Relaxometry)

#Config, specifc settings should be made their
include(config/config.cmake)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(SOURCE_WILDCARDS *.h *.H *.hpp *.hh *.hxx *.c *.C *.cpp *.cc *.cxx)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

IF(WIN32)
  set(GDCM_DIR "${PROJECT_SOURCE_DIR}/dependencies/gdcm_bin")
  set(GDCM_LIB gdcmCommon gdcmDICT gdcmDSED gdcmIOD gdcmMSFF gdcmcharls gdcmexpat gdcmgetopt gdcmjpeg12 gdcmjpeg16 gdcmjpeg8 gdcmopenjpeg gdcmzlib)
  link_directories("${PROJECT_SOURCE_DIR}/dependencies/gdcm_bin/Debug/bin/")
ELSE(WIN32)
  set(GDCM_LIB gdcmMSFF gdcmDSED gdcmCommon)
  set(NIFTI_LIB fslio niftiio znz z)
  include_directories("${PROJECT_SOURCE_DIR}/dependencies/nifticlib_bin/nifticlib/usr/local/include/nifti")
  link_directories("${PROJECT_SOURCE_DIR}/dependencies/nifticlib_bin/nifticlib/usr/local/lib")
ENDIF(WIN32)

#enable_testing()
#add_subdirectory(test)# EXCLUDE_FROM_ALL)

add_subdirectory(src)
