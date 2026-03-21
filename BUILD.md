# Building GauXC

This document provides concrete build instructions for GauXC in the `livdft`
conda environment on the current machine (4× A100 80GB, CUDA 13.1).

For general CMake options and usage as a library, see [README.md](README.md).

## Prerequisites

The `livdft` conda environment provides:

- **C/C++ compilers**: gcc/g++ 15.2 (conda cross-compilers)
- **BLAS**: OpenBLAS 0.3.30
- **MPI**: OpenMPI 4.1.6
- **HDF5**: 1.14.6
- **CUDA nvcc**: 13.1 (conda `cuda-nvcc` package)
- **PyTorch**: with CUDA 12 runtime (for OneDFT support)

CMake is installed via pip (`pip install cmake`).

Auto-fetched dependencies (via FetchContent — no manual install needed):
ExchCXX, IntegratorXX, Eigen3, Catch2, Gau2Grid, HighFive, nlohmann_json.

## Environment Setup

Activate the conda environment and set up the Torch cmake path:

```bash
micromamba activate livdft
export TORCH_CMAKE=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
```

For running tests, PyTorch pulls in CUDA 12 runtime libraries installed via pip.
Export their paths so the test executables can find them:

```bash
NVIDIA_BASE="$HOME/.local/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="\
$NVIDIA_BASE/cuda_runtime/lib:\
$NVIDIA_BASE/cuda_cupti/lib:\
$NVIDIA_BASE/cublas/lib:\
$NVIDIA_BASE/cuda_nvrtc/lib:\
$NVIDIA_BASE/cudnn/lib:\
$NVIDIA_BASE/cufft/lib:\
$NVIDIA_BASE/cusolver/lib:\
$NVIDIA_BASE/cusparse/lib:\
$NVIDIA_BASE/cusparselt/lib:\
$NVIDIA_BASE/nvjitlink/lib:\
$NVIDIA_BASE/nvtx/lib:\
$NVIDIA_BASE/cufile/lib:\
$NVIDIA_BASE/curand/lib:\
/usr/local/cuda-13.1/lib64:\
$LD_LIBRARY_PATH"
```

## CPU-Only Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGAUXC_ENABLE_CUDA=OFF \
  -DGAUXC_ENABLE_HIP=OFF \
  -DGAUXC_ENABLE_MPI=ON \
  -DGAUXC_ENABLE_OPENMP=ON \
  -DGAUXC_ENABLE_TESTS=ON \
  -DGAUXC_ENABLE_HDF5=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.1 \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$TORCH_CMAKE;/usr/local/cuda-13.1" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -Wno-dev

cmake --build build -j$(nproc)
```

### Run Tests

```bash
cd build && ctest --output-on-failure
```

## CUDA Build

Targets NVIDIA A100 GPUs (compute capability 8.0).

The conda CUDA toolkit provides `nvcc` and `cudart` but not cuBLAS headers.
These live in the system CUDA installation at `/usr/local/cuda-13.1/` and must
be added explicitly to both C++ and CUDA compiler include paths.

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGAUXC_ENABLE_CUDA=ON \
  -DGAUXC_ENABLE_HIP=OFF \
  -DGAUXC_ENABLE_MAGMA=OFF \
  -DGAUXC_ENABLE_NCCL=OFF \
  -DGAUXC_ENABLE_CUTLASS=OFF \
  -DGAUXC_ENABLE_MPI=ON \
  -DGAUXC_ENABLE_OPENMP=ON \
  -DGAUXC_ENABLE_TESTS=ON \
  -DGAUXC_ENABLE_HDF5=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.1 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.1 \
  -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-13.1/targets/x86_64-linux/include \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$TORCH_CMAKE;/usr/local/cuda-13.1" \
  -DCMAKE_CXX_FLAGS="-I/usr/local/cuda-13.1/targets/x86_64-linux/include" \
  -DCMAKE_CUDA_FLAGS="-I/usr/local/cuda-13.1/targets/x86_64-linux/include" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -Wno-dev

cmake --build build-cuda -j$(nproc)
```

### Run Tests

```bash
cd build-cuda && ctest --output-on-failure
```

## Troubleshooting

### `libcupti.so.12: cannot open shared object file`

PyTorch was built against CUDA 12 runtime libraries installed via pip.
Set `LD_LIBRARY_PATH` as shown in [Environment Setup](#environment-setup).

### `cublas_v2.h: No such file or directory`

The conda CUDA package does not include cuBLAS headers. Ensure both
`CMAKE_CXX_FLAGS` and `CMAKE_CUDA_FLAGS` include
`-I/usr/local/cuda-13.1/targets/x86_64-linux/include`.

### `Compatibility with CMake < 3.5 has been removed`

The HighFive dependency uses an old `cmake_minimum_required`. Add
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to the configure command.

### OneDFT MPI test failures (values 2× reference)

The OneDFT test case reports values exactly double the expected reference when
run under MPI. This is a known issue in the OneDFT reduction path, not a build
problem. All other tests pass.
