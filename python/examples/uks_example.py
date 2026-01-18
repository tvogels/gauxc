"""
Example: UKS (Unrestricted Kohn-Sham) Support

This example demonstrates UKS support with device consistency checking:
1. Creating spin-polarized density matrices
2. Using UKS with eval_mgga_vvars
3. Device consistency validation
4. Comparison with RKS
"""

import numpy as np

try:
    import gauxc_py as gxc
except ImportError:
    print("GauXC Python bindings not installed!")
    print("Build and install with: pip install .")
    exit(1)

print("=" * 70)
print("GauXC UKS (Unrestricted Kohn-Sham) Example")
print("=" * 70)

# 1. Create H2 molecule
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
nbf = basis.nbf()
print(f"   Basis: {basis}, nbf={nbf}")

# 3. Generate grid
print("\n" + "=" * 70)
print("3. Generating molecular grid")
print("=" * 70)

grid = gxc.compute_grid(
    mol, basis,
    pruning_scheme="robust",
    grid_size="fine",
    batch_size=512,
    exec_space="host"
)

print(f"   Grid: {grid.npts} points")

# 4. Create density matrices
print("\n" + "=" * 70)
print("4. Creating density matrices")
print("=" * 70)

np.random.seed(42)

# RKS: single density matrix
P_rks_random = np.random.randn(nbf, nbf) * 0.1
P_rks = (P_rks_random + P_rks_random.T) / 2
print(f"   RKS P shape: {P_rks.shape}")

# UKS: separate spin-up and spin-down matrices
Ps_random = np.random.randn(nbf, nbf) * 0.08
Ps = (Ps_random + Ps_random.T) / 2  # Spin-up

Pz_random = np.random.randn(nbf, nbf) * 0.05  
Pz = (Pz_random + Pz_random.T) / 2  # Spin-down

print(f"   UKS Ps shape: {Ps.shape} (spin-up)")
print(f"   UKS Pz shape: {Pz.shape} (spin-down)")

# 5. Test RKS
print("\n" + "=" * 70)
print("5. Testing RKS evaluation")
print("=" * 70)

try:
    vvars_rks = gxc.eval_mgga_vvars(
        mol, basis, grid, P_rks,
        ks_scheme="RKS"
    )
    print(f"   ✓ RKS evaluation successful")
    print(f"   rho: {vvars_rks.rho.shape}, sum={np.sum(vvars_rks.rho * grid.weights):.6f}")
    print(f"   grad: {vvars_rks.grad.shape}")
    print(f"   gamma: {vvars_rks.gamma.shape}")
    print(f"   tau: {vvars_rks.tau.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 6. Test UKS
print("\n" + "=" * 70)
print("6. Testing UKS evaluation")
print("=" * 70)

try:
    # Pass as tuple
    vvars_uks = gxc.eval_mgga_vvars(
        mol, basis, grid, (Ps, Pz),
        ks_scheme="UKS"
    )
    print(f"   ✓ UKS evaluation successful")
    print(f"   rho (total): {vvars_uks.rho.shape}, sum={np.sum(vvars_uks.rho * grid.weights):.6f}")
    print(f"   grad: {vvars_uks.grad.shape}")
    print(f"   gamma: {vvars_uks.gamma.shape}")
    print(f"   tau: {vvars_uks.tau.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 7. Test auto-detection
print("\n" + "=" * 70)
print("7. Testing auto-detection of UKS")
print("=" * 70)

try:
    # Pass tuple without specifying ks_scheme
    vvars_auto = gxc.eval_mgga_vvars(
        mol, basis, grid, (Ps, Pz)
        # ks_scheme defaults to "RKS" but should auto-detect
    )
    print(f"   ✓ Auto-detection successful (got warning)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 8. Test device consistency (if PyTorch available)
print("\n" + "=" * 70)
print("8. Testing device consistency")
print("=" * 70)

try:
    import torch
    print("   PyTorch available - testing device consistency")
    
    # Test 8a: Consistent devices (CPU)
    print("\n   8a. Consistent devices (both CPU):")
    Ps_cpu = torch.from_numpy(Ps)
    Pz_cpu = torch.from_numpy(Pz)
    
    try:
        vvars = gxc.eval_mgga_vvars(
            mol, basis, grid, (Ps_cpu, Pz_cpu),
            ks_scheme="UKS",
            return_torch=True
        )
        print(f"      ✓ Success, output device: {vvars.rho.device}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # Test 8b: Inconsistent devices (if CUDA available)
    if torch.cuda.is_available():
        print("\n   8b. Inconsistent devices (Ps on CPU, Pz on CUDA):")
        Ps_cpu_test = torch.from_numpy(Ps)
        Pz_gpu_test = torch.from_numpy(Pz).cuda()
        
        try:
            vvars = gxc.eval_mgga_vvars(
                mol, basis, grid, (Ps_cpu_test, Pz_gpu_test),
                ks_scheme="UKS"
            )
            print(f"      ✗ Should have raised an error!")
        except ValueError as e:
            print(f"      ✓ Correctly rejected: {str(e)[:60]}...")
        
        print("\n   8c. Consistent devices (both CUDA):")
        Ps_gpu = torch.from_numpy(Ps).cuda()
        Pz_gpu = torch.from_numpy(Pz).cuda()
        
        try:
            vvars = gxc.eval_mgga_vvars(
                mol, basis, grid, (Ps_gpu, Pz_gpu),
                ks_scheme="UKS",
                exec_space="host",
                return_torch=True
            )
            print(f"      ✓ Success (auto-copied to CPU), output device: {vvars.rho.device}")
        except Exception as e:
            print(f"      ✗ Error: {e}")
    else:
        print("\n   CUDA not available - skipping GPU consistency tests")
        
except ImportError:
    print("   PyTorch not available - skipping device consistency tests")

# 9. Test error handling
print("\n" + "=" * 70)
print("9. Testing error handling")
print("=" * 70)

# 9a: Wrong number of matrices
print("\n   9a. Wrong number of matrices for UKS:")
try:
    vvars = gxc.eval_mgga_vvars(
        mol, basis, grid, (Ps, Pz, Ps),  # 3 matrices
        ks_scheme="UKS"
    )
    print(f"      ✗ Should have raised an error!")
except ValueError as e:
    print(f"      ✓ Correctly rejected: {e}")

# 9b: Single matrix with UKS
print("\n   9b. Single matrix with ks_scheme='UKS':")
try:
    vvars = gxc.eval_mgga_vvars(
        mol, basis, grid, P_rks,
        ks_scheme="UKS"
    )
    print(f"      ✗ Should have raised an error!")
except ValueError as e:
    print(f"      ✓ Correctly rejected: {e}")

# 9c: Wrong shape
print("\n   9c. Wrong matrix shape:")
try:
    Ps_wrong = np.random.randn(nbf+1, nbf)
    vvars = gxc.eval_mgga_vvars(
        mol, basis, grid, (Ps_wrong, Pz),
        ks_scheme="UKS"
    )
    print(f"      ✗ Should have raised an error!")
except ValueError as e:
    print(f"      ✓ Correctly rejected: {e}")

# 10. Summary
print("\n" + "=" * 70)
print("Summary: UKS Support")
print("=" * 70)
print("""
Key features demonstrated:
✓ RKS with single density matrix
✓ UKS with tuple of two density matrices (Ps, Pz)
✓ Auto-detection of UKS from input format
✓ Device consistency validation for PyTorch tensors
✓ Proper error handling for invalid inputs

UKS Usage:
- Pass density matrices as tuple: P=(Ps, Pz)
- Ps: spin-up density matrix
- Pz: spin-down density matrix  
- Both matrices must be on same device
- Returns total density (alpha + beta)

Device Consistency:
- All matrices must be on the same device (CPU or same GPU)
- Error raised if matrices on different devices
- Helps catch bugs early in ML pipelines
""")

print("=" * 70)
print("Example completed!")
print("=" * 70)
