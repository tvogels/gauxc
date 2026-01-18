#!/bin/bash
# Test script for building GauXC Python bindings

set -e

echo "================================================"
echo "GauXC Python Bindings Build Test"
echo "================================================"

# Check prerequisites
echo ""
echo "1. Checking prerequisites..."
echo "   - CMake version:"
cmake --version | head -1
echo "   - Python version:"
python3 --version
echo "   - C++ compiler:"
g++ --version | head -1

# Check Python packages
echo ""
echo "2. Checking Python packages..."
python3 -c "import numpy; print('   ✓ NumPy:', numpy.__version__)"
python3 -c "import pybind11; print('   ✓ pybind11:', pybind11.__version__)"

# Try to install if not found
if ! python3 -c "import scikit_build_core" 2>/dev/null; then
    echo "   Installing scikit-build-core..."
    pip3 install scikit-build-core
fi
python3 -c "import scikit_build_core; print('   ✓ scikit-build-core installed')"

# Configure build
echo ""
echo "3. Configuring CMake build..."
BUILD_DIR="/tmp/gauxc_test_build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DGAUXC_ENABLE_PYTHON=ON \
      -DGAUXC_ENABLE_TESTS=OFF \
      -DGAUXC_ENABLE_MPI=OFF \
      -DGAUXC_ENABLE_HDF5=OFF \
      /path/to/gauxc

# Build
echo ""
echo "4. Building..."
cmake --build . -j$(nproc)

# Install
echo ""
echo "5. Installing Python package..."
pip3 install -e /path/to/gauxc

# Test import
echo ""
echo "6. Testing import..."
python3 -c "import gauxc_py; print('   ✓ Successfully imported gauxc_py')"

echo ""
echo "================================================"
echo "Build test completed successfully!"
echo "================================================"
