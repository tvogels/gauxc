"""
Example: GPU Tensor Support with GauXC Python bindings

This example demonstrates GPU tensor handling with CUDA/HIP support:
1. Creating molecules and grids
2. Using GPU tensors with eval_mgga_vvars
3. Automatic device detection and handling
4. Zero-copy operations where possible
"""

import numpy as np

try:
    import gauxc_py as gxc
except ImportError:
    print("GauXC Python bindings not installed!")
    print("Build and install with: pip install .")
    exit(1)

print("=" * 70)
print("GauXC GPU Tensor Support Example")
print("=" * 70)

# Check if PyTorch is available
try:
    import torch
    print(f"\n✓ PyTorch {torch.__version__} available")
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("✗ CUDA not available - will demonstrate CPU operations")
except ImportError:
    print("\n✗ PyTorch not installed - this example requires PyTorch")
    exit(1)

# 1. Create H2 molecule (simple example)
print("\n" + "=" * 70)
print("1. Creating H2 molecule")
print("=" * 70)

atomic_numbers = np.array([1, 1], dtype=np.int32)
coords = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.4],  # 1.4 bohr
], dtype=np.float64)

mol = gxc.molecule_from_arrays(atomic_numbers, coords)
print(f"   Created: {mol}")

# 2. Create minimal basis set
print("\n" + "=" * 70)
print("2. Creating basis set (STO-3G-like)")
print("=" * 70)

basis = gxc.BasisSet()
alphas = np.array([3.4253, 0.6239, 0.1689], dtype=np.float64)
coeffs = np.array([0.1543, 0.5353, 0.4446], dtype=np.float64)

for i in range(2):
    shell = gxc.create_simple_shell(
        1, 0, alphas, coeffs,
        coords[i, 0], coords[i, 1], coords[i, 2],
        spherical=True
    )
    basis.append(shell)

basis.set_basis_tolerance(1e-10)
print(f"   Basis: {basis}")

# 3. Generate grid
print("\n" + "=" * 70)
print("3. Generating molecular grid")
print("=" * 70)

grid = gxc.compute_grid(
    mol, basis,
    pruning_scheme="robust",
    grid_size="fine",  # Use fine for faster demo
    batch_size=512,
    exec_space="host"
)

print(f"   Grid: {grid.npts} points")

# 4. Create density matrix
print("\n" + "=" * 70)
print("4. Creating test density matrix")
print("=" * 70)

nbf = basis.nbf()
np.random.seed(42)
P_random = np.random.randn(nbf, nbf) * 0.1
P_cpu = (P_random + P_random.T) / 2  # Symmetric

print(f"   Density matrix shape: {P_cpu.shape}")
print(f"   Dtype: {P_cpu.dtype}")

# 5. Test with CPU tensors
print("\n" + "=" * 70)
print("5. Testing with CPU tensors")
print("=" * 70)

P_cpu_torch = torch.from_numpy(P_cpu)
print(f"   Input tensor: {P_cpu_torch.shape}, device={P_cpu_torch.device}")

try:
    vvars_cpu = gxc.eval_mgga_vvars(
        mol, basis, grid, P_cpu_torch,
        exec_space="host",
        return_torch=True
    )
    print(f"   ✓ Evaluation successful")
    print(f"   Output rho: {vvars_cpu.rho.shape}, device={vvars_cpu.rho.device}")
    print(f"   Output grad: {vvars_cpu.grad.shape}, device={vvars_cpu.grad.device}")
    print(f"   Sample rho values: {vvars_cpu.rho[:5]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 6. Test with GPU tensors (if available)
if has_cuda:
    print("\n" + "=" * 70)
    print("6. Testing with GPU tensors (CUDA)")
    print("=" * 70)
    
    # Move tensor to GPU
    P_gpu = P_cpu_torch.cuda()
    print(f"   Input tensor: {P_gpu.shape}, device={P_gpu.device}")
    
    # Test with host execution (should auto-copy to CPU)
    print("\n   6a. GPU tensor with host execution:")
    try:
        vvars_host = gxc.eval_mgga_vvars(
            mol, basis, grid, P_gpu,
            exec_space="host",
            return_torch=True
        )
        print(f"      ✓ Evaluation successful (auto-copied to CPU)")
        print(f"      Output rho device: {vvars_host.rho.device}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # Test with device execution (requires GauXC built with CUDA)
    print("\n   6b. GPU tensor with device execution:")
    try:
        vvars_device = gxc.eval_mgga_vvars(
            mol, basis, grid, P_gpu,
            exec_space="device",
            return_torch=True,
            device='cuda'  # Return on same GPU
        )
        print(f"      ✓ Evaluation successful")
        print(f"      Output rho device: {vvars_device.rho.device}")
        print(f"      Sample rho values: {vvars_device.rho[:5]}")
    except RuntimeError as e:
        if "not built with CUDA" in str(e):
            print(f"      ⚠ Skipped: GauXC not built with CUDA support")
            print(f"        To enable, rebuild with -DGAUXC_ENABLE_CUDA=ON")
        else:
            print(f"      ✗ Error: {e}")
    except ValueError as e:
        print(f"      ⚠ {e}")
    except Exception as e:
        print(f"      ✗ Unexpected error: {e}")
    
    # Test device conversion
    print("\n   6c. Converting output to specific GPU:")
    try:
        vvars_cpu_out = gxc.eval_mgga_vvars(
            mol, basis, grid, P_cpu_torch,
            exec_space="host",
            return_torch=True,
            device='cuda:0'  # Move output to GPU
        )
        print(f"      ✓ Converted successfully")
        print(f"      Output rho device: {vvars_cpu_out.rho.device}")
    except Exception as e:
        print(f"      ✗ Error: {e}")

else:
    print("\n" + "=" * 70)
    print("6. GPU testing skipped (CUDA not available)")
    print("=" * 70)
    print("   To test GPU functionality:")
    print("   1. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("   2. Build GauXC with CUDA: cmake -DGAUXC_ENABLE_CUDA=ON ...")
    print("   3. Ensure CUDA drivers and runtime are installed")

# 7. Summary
print("\n" + "=" * 70)
print("Summary: GPU Tensor Support")
print("=" * 70)
print("""
Key features demonstrated:
✓ CPU tensors work seamlessly
✓ GPU tensors auto-detected and handled
✓ Automatic CPU/GPU transfer based on exec_space
✓ Can specify output device with return_torch + device parameter
✓ Error messages guide users on build requirements

Behavior:
- exec_space='host': Works with any tensor, GPU tensors copied to CPU
- exec_space='device': Requires GPU tensor + GauXC built with CUDA/HIP
- return_torch=True: Returns PyTorch tensors
- device parameter: Controls output tensor device placement

Notes:
- Current implementation copies GPU tensors to CPU for computation
- Future enhancement: Native GPU pointer support for zero-copy operations
""")

print("=" * 70)
print("Example completed!")
print("=" * 70)
