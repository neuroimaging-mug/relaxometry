cmake_minimum_required(VERSION 2.8)

# Build nifti
add_subdirectory(../nifticlib nifticlib)

# Get needed variables
get_directory_property(NIFTI_INSTALL_INCLUDE_DIR DIRECTORY ../nifticlib DEFINITION NIFTI_INSTALL_INCLUDE_DIR)
get_directory_property(NIFTI_INSTALL_LIB_DIR     DIRECTORY ../nifticlib DEFINITION NIFTI_INSTALL_LIB_DIR)

# Set needed variables
set(NIFTI_INSTALL_PACKAGE_DIR ${NIFTI_INSTALL_LIB_DIR}/nifticlib-2.0.0)

# Create config and use-file
add_subdirectory(configuration configuration)
