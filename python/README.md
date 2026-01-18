# GauXC Python Bindings

Python bindings for GauXC - GPU-accelerated exchange-correlation library with PyTorch integration.

## Features

- **Python API** for GauXC core functionality:
  - Molecule construction
  - Gaussian basis set specification
  - Molecular grid generation with configurable parameters
  - Meta-GGA variable evaluation (ρ, ∇ρ, γ, τ, ∇²ρ)

- **PyTorch Integration**:
  - Zero-copy tensor conversion via DLPack
  - Support for both CPU and GPU tensors (when built with CUDA/HIP)
  - Seamless integration with PyTorch workflows

- **Clean API Design**:
  - Pythonic interface with dataclasses
  - Type hints and documentation
  - Both numpy arrays and torch tensors supported

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/wavefunction91/GauXC.git
cd GauXC

# Install with pip (builds C++ extension)
pip install .

# Or install in development mode
pip install -e .

# With PyTorch support
pip install ".[torch]"
```

### Build Options

The Python bindings can be configured via CMake options or environment variables:

```bash
# Enable Python bindings
cmake -DGAUXC_ENABLE_PYTHON=ON ...

# With CUDA support
cmake -DGAUXC_ENABLE_PYTHON=ON -DGAUXC_ENABLE_CUDA=ON ...
```

## Quick Start

```python
import numpy as np
import gauxc_py as gxc

# Create H2O molecule
atomic_numbers = np.array([8, 1, 1])
coords = np.array([
    [0.0000,  0.0000,  0.1173],
    [0.0000,  0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692],
])
mol = gxc.molecule_from_arrays(atomic_numbers, coords)

# Create basis set (simplified)
basis = gxc.BasisSet()
# ... add shells to basis ...

# Generate molecular grid
grid = gxc.compute_grid(
    mol, basis,
    pruning_scheme="robust",
    batch_size=512,
    grid_size="ultrafine"
)

print(f"Grid has {grid.npts} points")
print(f"Points shape: {grid.points.shape}")  # (npts, 3)
print(f"Weights shape: {grid.weights.shape}")  # (npts,)

# Create density matrix
nbf = basis.nbf()
P = np.random.randn(nbf, nbf)
P = (P + P.T) / 2  # Make symmetric

# Evaluate meta-GGA variables
vvars = gxc.eval_mgga_vvars(
    mol, basis, grid, P,
    ks_scheme='RKS',
    need_lapl=False
)

print(f"rho shape: {vvars.rho.shape}")      # (npts,)
print(f"grad shape: {vvars.grad.shape}")    # (npts, 3)
print(f"gamma shape: {vvars.gamma.shape}")  # (npts,)
print(f"tau shape: {vvars.tau.shape}")      # (npts,)
```

## PyTorch Integration

### Basic Usage

```python
import torch
import gauxc_py as gxc

# Create molecule from torch tensors
atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.int32)
coords = torch.tensor([
    [0.0000,  0.0000,  0.1173],
    [0.0000,  0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692],
], dtype=torch.float64)

mol = gxc.molecule_from_arrays(atomic_numbers, coords)

# ... (basis and grid generation as before) ...

# Use torch density matrix
P_torch = torch.randn(nbf, nbf, dtype=torch.float64)
P_torch = (P_torch + P_torch.T) / 2

# Evaluate and get torch tensors back
vvars = gxc.eval_mgga_vvars(
    mol, basis, grid, P_torch,
    return_torch=True
)

# Now vvars contains torch tensors
assert isinstance(vvars.rho, torch.Tensor)
```

### GPU Tensor Support (CUDA/HIP)

When GauXC is built with CUDA or HIP support, GPU tensors are fully supported:

```python
import torch
import gauxc_py as gxc

# Move density matrix to GPU
P_gpu = P_torch.cuda()

# Option 1: Host execution (auto-copies GPU tensor to CPU)
vvars_cpu = gxc.eval_mgga_vvars(
    mol, basis, grid, P_gpu,
    exec_space="host",
    return_torch=True
)
# Output on CPU: vvars_cpu.rho.device = 'cpu'

# Option 2: Device execution (requires CUDA/HIP build)
vvars_gpu = gxc.eval_mgga_vvars(
    mol, basis, grid, P_gpu,
    exec_space="device",
    return_torch=True,
    device='cuda'  # Return results on GPU
)
# Output on GPU: vvars_gpu.rho.device = 'cuda:0'

# Option 3: CPU input, GPU output
vvars_to_gpu = gxc.eval_mgga_vvars(
    mol, basis, grid, P_torch,  # CPU tensor
    exec_space="host",
    return_torch=True,
    device='cuda:0'  # Move output to GPU
)
```

**Behavior:**
- **CPU tensors**: Work with any build (host or device execution)
- **GPU tensors with `exec_space="host"`**: Auto-copied to CPU, computation on CPU
- **GPU tensors with `exec_space="device"`**: Requires CUDA/HIP build, computation on GPU
- **`device` parameter**: Controls output tensor placement when `return_torch=True`

**Notes:**
- Current implementation copies GPU tensors to CPU for computation
- Zero-copy GPU operations planned for future release
- Error messages guide users if CUDA/HIP build is required

See `examples/gpu_tensor_example.py` for a complete demonstration.

## API Reference

### Core Classes

- `Molecule`: Container for atoms and coordinates
- `Atom`: Individual atom with atomic number and position
- `BasisSet`: Collection of Gaussian basis shells
- `Shell`: Single GTO shell with primitives

### Grid Generation

- `compute_grid()`: Generate molecular integration grid
- `GridData`: Container for grid points and weights
- Grid parameters:
  - `pruning_scheme`: 'unpruned', 'robust', 'treutler'
  - `grid_size`: 'fine', 'ultrafine', 'superfine', 'gm3', 'gm5'
  - `radial_quad`: 'becke', 'mura_knowles', 'treutler_ahlrichs', etc.

### Variable Evaluation

- `eval_mgga_vvars()`: Evaluate meta-GGA variables on grid
- `MGGAVariables`: Container for computed variables
  - `rho`: Electron density ρ(r)
  - `grad`: Density gradient ∇ρ(r)
  - `gamma`: Contracted gradient γ(r) = |∇ρ(r)|²
  - `tau`: Kinetic energy density τ(r)
  - `lapl`: Laplacian ∇²ρ(r) (optional)

## Examples

See the `examples/` directory for complete examples:
- `h2o_example.py`: Basic H2O molecule example

## Testing

```bash
# Run tests
pytest python/tests/

# Run with verbose output
pytest python/tests/ -v

# Run specific test
pytest python/tests/test_basic.py::TestMolecule
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- CMake >= 3.20
- C++17 compiler
- pybind11 >= 2.10.0

### Optional

- PyTorch >= 1.9.0 (for tensor integration)
- pytest >= 6.0 (for testing)

## Notes

- The bindings currently support CPU execution; GPU support requires building with CUDA/HIP enabled
- Grid generation uses the same high-quality quadratures as the C++ library
- Variable evaluation leverages optimized C++ kernels for performance

## License

GauXC is made freely available under the terms of a modified 3-Clause BSD license. See LICENSE.txt for details.

## Citation

If you use GauXC in your research, please cite the relevant publications listed in the main README.md.
