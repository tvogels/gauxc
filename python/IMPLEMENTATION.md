# GauXC Python Bindings - Implementation Notes

## Overview

This document describes the Python/PyTorch bindings added to GauXC for GPU-accelerated exchange-correlation calculations.

## Architecture

### C++ Binding Layer (`python/src/`)

The C++ bindings use **pybind11** to expose GauXC functionality to Python:

1. **module.cxx**: Main module entry point that registers all submodules
2. **molecule.cxx**: Bindings for `Molecule`, `Atom`, and `AtomicNumber` classes
3. **basisset.cxx**: Bindings for `BasisSet` and `Shell` classes  
4. **grid.cxx**: Bindings for grid generation (`MolGrid`, `Grid`, grid extraction)
5. **integrator.cxx**: Stub for meta-GGA variable evaluation (placeholder for future)
6. **torch_utils.cxx**: PyTorch integration utilities

### Python API Layer (`python/gauxc_py/`)

Python wrappers provide a clean, Pythonic interface:

1. **core.py**: Re-exports C++ bindings with convenience functions
2. **grid_utils.py**: High-level grid generation API with `GridData` dataclass
3. **vvar_eval.py**: Meta-GGA variable evaluation with `MGGAVariables` dataclass
4. **__init__.py**: Package initialization and public API

### Build System

- **pyproject.toml**: Modern Python packaging using `scikit-build-core`
- **cmake/gauxc-pybind11.cmake**: CMake configuration for fetching pybind11
- **python/CMakeLists.txt**: Python module build configuration
- **CMakeLists.txt** (root): Added `GAUXC_ENABLE_PYTHON` option

## Key Design Decisions

### 1. Grid Extraction Strategy

Since GauXC's internal grid structure is complex (batched, distributed), we extract grid data through the LoadBalancer:

```cpp
// Create LoadBalancer with molecular weights applied
LoadBalancer lb = ...;
MolecularWeights mw = ...;
mw.modify_weights(lb);

// Extract points and weights from tasks
for (const auto& task : lb->get_tasks()) {
    for (size_t i = 0; i < task.npts; i++) {
        points[offset + i] = task.points[i];
        weights[offset + i] = task.weights[i];
    }
}
```

This gives us the final molecular grid with partition weights applied, ready for integration.

### 2. MPI Handling

Python bindings default to single-process mode for simplicity:
- Uses `MPI_COMM_SELF` when MPI is enabled in the build
- Users can still use GauXC's MPI capabilities from C++ if needed
- Simplifies Python API while maintaining full C++ functionality

### 3. PyTorch Integration

Zero-copy tensor conversion via DLPack protocol (Python-side):

```python
# NumPy array
grid_data = gxc.compute_grid(mol, basis)
points_np = grid_data.points  # numpy.ndarray

# Convert to PyTorch (zero-copy if contiguous)
import torch
points_torch = torch.from_numpy(points_np)

# Or directly
grid_data_torch = grid_data.to_torch()
```

This avoids copying data and maintains interoperability.

### 4. Meta-GGA Variables (Stub Implementation)

The `eval_mgga_vvars` function currently returns zeros as a placeholder. Full implementation requires:

1. **Collocation Evaluation**: Call `LocalWorkDriver::eval_collocation*` methods to evaluate basis functions and derivatives on grid points

2. **Density Matrix Contraction**: Compute density variables from collocation and density matrix:
   ```
   rho(r) = sum_mu,nu P_mu,nu * phi_mu(r) * phi_nu(r)
   grad_rho(r) = sum_mu,nu P_mu,nu * [grad_phi_mu(r) * phi_nu(r) + ...]
   tau(r) = sum_mu,nu P_mu,nu * grad_phi_mu(r) · grad_phi_nu(r) / 2
   ```

3. **Memory Management**: Allocate scratch space for collocation evaluation on each task

The infrastructure is in place; the implementation requires exposing more of GauXC's internal collocation APIs.

## API Examples

### Creating a Molecule

```python
import numpy as np
import gauxc_py as gxc

# From arrays
z = np.array([8, 1, 1])  # O, H, H
coords = np.array([[0.0, 0.0, 0.1], [0.0, 0.8, -0.5], [0.0, -0.8, -0.5]])
mol = gxc.molecule_from_arrays(z, coords)

# Or manually
mol = gxc.Molecule()
mol.append(gxc.Atom(gxc.AtomicNumber(8), 0.0, 0.0, 0.1))
# ...
```

### Creating a Basis Set

```python
# Create shells
alphas = np.array([3.425, 0.624, 0.169])
coeffs = np.array([0.154, 0.535, 0.445])

shell = gxc.create_simple_shell(
    Z=1, l=0, alphas=alphas, coeffs=coeffs,
    x=0.0, y=0.0, z=0.0, spherical=True
)

basis = gxc.BasisSet()
basis.append(shell)
basis.set_basis_tolerance(1e-10)
```

### Generating a Grid

```python
grid = gxc.compute_grid(
    mol, basis,
    pruning_scheme="robust",      # or "unpruned", "treutler"
    batch_size=512,
    radial_quad="mura_knowles",   # or "becke", "treutler_ahlrichs"
    grid_size="ultrafine",        # or "fine", "superfine", "gm3", "gm5"
    exec_space="host"             # or "device" if CUDA/HIP enabled
)

print(f"Grid has {grid.npts} points")
print(f"Points shape: {grid.points.shape}")  # (npts, 3)
print(f"Weights shape: {grid.weights.shape}")  # (npts,)
```

### Evaluating Variables (Stub)

```python
# Create density matrix
nbf = basis.nbf()
P = np.random.randn(nbf, nbf)
P = (P + P.T) / 2  # Symmetric

# Evaluate (currently returns zeros)
vvars = gxc.eval_mgga_vvars(mol, basis, grid, P, ks_scheme='RKS')

print(vvars.rho.shape)    # (npts,)
print(vvars.grad.shape)   # (npts, 3)
print(vvars.gamma.shape)  # (npts,)
print(vvars.tau.shape)    # (npts,)
```

## Build Instructions

### Prerequisites

- Python >= 3.8
- NumPy >= 1.20
- CMake >= 3.20
- C++17 compiler
- pybind11 >= 2.10 (auto-fetched)
- GauXC dependencies (BLAS, ExchCXX, IntegratorXX)

### Building

```bash
# Configure with Python enabled
cmake -DGAUXC_ENABLE_PYTHON=ON \
      -DGAUXC_ENABLE_MPI=OFF \
      -DGAUXC_ENABLE_TESTS=OFF \
      <source_dir>

# Build
cmake --build . -j

# Install Python package
pip install -e <source_dir>
```

### Using pip (skips CMake if scikit-build-core is installed)

```bash
# From source directory
pip install .

# Development mode
pip install -e .

# With PyTorch
pip install ".[torch]"
```

## Testing

```bash
# Run Python tests
pytest python/tests/

# Run example
python python/examples/h2o_example.py
```

## Current Limitations

1. **Meta-GGA evaluation is a stub**: Returns zeros; requires exposing collocation APIs
2. **Single-process only**: Python API uses single MPI rank
3. **No device memory management**: GPU tensors require explicit device placement
4. **Limited functional support**: Future work to expose XC functional selection

## Future Enhancements

1. **Complete vvar evaluation**: Implement full collocation + density evaluation
2. **Functional selection**: Expose ExchCXX functional library to Python
3. **XC energy/potential**: Bind `XCIntegrator::eval_exc_vxc` methods
4. **GPU support**: Direct GPU tensor input/output when CUDA/HIP enabled
5. **Batch processing**: Expose batch iteration for large grids
6. **Custom quadratures**: Allow users to specify custom atomic grids

## Files Added

```
gauxc/
├── pyproject.toml                    # Python packaging config
├── cmake/gauxc-pybind11.cmake        # CMake pybind11 integration
├── CMakeLists.txt                    # Modified to add GAUXC_ENABLE_PYTHON
├── README.md                         # Modified to mention Python bindings
├── python/
│   ├── CMakeLists.txt                # Python module build
│   ├── README.md                     # Python bindings documentation
│   ├── test_build.sh                 # Build test script
│   ├── src/
│   │   ├── module.cxx                # Main pybind11 module
│   │   ├── molecule.cxx              # Molecule bindings
│   │   ├── basisset.cxx              # BasisSet bindings
│   │   ├── grid.cxx                  # Grid bindings
│   │   ├── integrator.cxx            # Integrator stub
│   │   └── torch_utils.cxx           # PyTorch utilities
│   ├── gauxc_py/
│   │   ├── __init__.py               # Package init
│   │   ├── core.py                   # Core wrappers
│   │   ├── grid_utils.py             # Grid utilities
│   │   └── vvar_eval.py              # Variable evaluation
│   ├── tests/
│   │   └── test_basic.py             # Unit tests
│   └── examples/
│       └── h2o_example.py            # H2O example
```

## References

- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [scikit-build-core](https://scikit-build-core.readthedocs.io/)
- [PyTorch DLPack](https://pytorch.org/docs/stable/dlpack.html)
- [GauXC Paper](https://doi.org/10.3389/fchem.2020.581058)
