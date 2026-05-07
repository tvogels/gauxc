#
# GauXC Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy).
#
# (c) 2024-2025, Microsoft Corporation
#
# All rights reserved.
#
# See LICENSE.txt for details
#

message(STATUS "Building GauXC Python Bindings")

# Include pybind11
include(gauxc-pybind11)

# Create the Python module
pybind11_add_module(_gauxc_core MODULE
  src/module.cxx
  src/molecule.cxx
  src/basisset.cxx
  src/grid.cxx
  src/integrator.cxx
  src/torch_utils.cxx
)

target_link_libraries(_gauxc_core PRIVATE gauxc::gauxc)

# Set output directory
set_target_properties(_gauxc_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/gauxc_py"
)

# Install the module
install(TARGETS _gauxc_core
  LIBRARY DESTINATION gauxc_py
)

# Install Python package files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gauxc_py/
  DESTINATION gauxc_py
  FILES_MATCHING PATTERN "*.py"
)
