# Build the configuration file for external projects using NIFTI and cmake
configure_file(
  NIFTIConfig.cmake.in
  ${NIFTI_BINARY_DIR}/NIFTIConfig.cmake
  @ONLY
)
install(
  FILES ${NIFTI_BINARY_DIR}/NIFTIConfig.cmake
  DESTINATION ${NIFTI_INSTALL_PACKAGE_DIR}
  COMPONENT Headers
)
configure_file(
  UseNIFTI.cmake.in
  ${NIFTI_BINARY_DIR}/UseNIFTI.cmake
  @ONLY
)
install(
  FILES ${NIFTI_BINARY_DIR}/UseNIFTI.cmake
  DESTINATION ${NIFTI_INSTALL_PACKAGE_DIR}
  COMPONENT Headers
)
