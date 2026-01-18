"""
Example: H2O molecule with GauXC Python bindings

This example demonstrates:
1. Creating a molecule (H2O)
2. Defining a simple basis set
3. Generating a molecular grid
4. Forming a density matrix
5. Evaluating meta-GGA variables
"""

import numpy as np
try:
    import gauxc_py as gxc
except ImportError:
    print("GauXC Python bindings not installed!")
    print("Build and install with: pip install .")
    exit(1)

print("=" * 60)
print("GauXC Python Bindings Example: H2O Molecule")
print("=" * 60)

# 1. Create H2O molecule
# Coordinates in Angstroms (optimized geometry)
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)  # O, H, H
coords = np.array([
    [0.0000,  0.0000,  0.1173],  # O
    [0.0000,  0.7572, -0.4692],  # H
    [0.0000, -0.7572, -0.4692],  # H
], dtype=np.float64)

mol = gxc.molecule_from_arrays(atomic_numbers, coords)
print(f"\n1. Created molecule: {mol}")
print(f"   Number of atoms: {mol.natoms()}")
print(f"   Max atomic number: {mol.maxZ().get()}")

# 2. Create a simple basis set (STO-3G-like for demonstration)
# This is a minimal basis for illustration
print("\n2. Creating simple basis set...")
basis = gxc.BasisSet()

# Oxygen 1s shell (approximate STO-3G)
alphas_O_1s = np.array([130.7093, 23.8089, 6.4436], dtype=np.float64)
coeffs_O_1s = np.array([0.1543, 0.5353, 0.4446], dtype=np.float64)
shell_O_1s = gxc.create_simple_shell(
    8, 0, alphas_O_1s, coeffs_O_1s, 
    coords[0, 0], coords[0, 1], coords[0, 2], spherical=True
)
basis.append(shell_O_1s)

# Oxygen 2s shell
alphas_O_2s = np.array([5.0331, 1.1695, 0.3803], dtype=np.float64)
coeffs_O_2s = np.array([-0.1000, 0.3995, 0.7001], dtype=np.float64)
shell_O_2s = gxc.create_simple_shell(
    8, 0, alphas_O_2s, coeffs_O_2s,
    coords[0, 0], coords[0, 1], coords[0, 2], spherical=True
)
basis.append(shell_O_2s)

# Oxygen 2p shell
alphas_O_2p = np.array([5.0331, 1.1695, 0.3803], dtype=np.float64)
coeffs_O_2p = np.array([0.1559, 0.6077, 0.3920], dtype=np.float64)
shell_O_2p = gxc.create_simple_shell(
    8, 1, alphas_O_2p, coeffs_O_2p,
    coords[0, 0], coords[0, 1], coords[0, 2], spherical=True
)
basis.append(shell_O_2p)

# Hydrogen shells (for both atoms)
alphas_H = np.array([3.4253, 0.6239, 0.1689], dtype=np.float64)
coeffs_H = np.array([0.1543, 0.5353, 0.4446], dtype=np.float64)

for i in [1, 2]:  # Two hydrogen atoms
    shell_H = gxc.create_simple_shell(
        1, 0, alphas_H, coeffs_H,
        coords[i, 0], coords[i, 1], coords[i, 2], spherical=True
    )
    basis.append(shell_H)

print(f"   Basis set: {basis}")
print(f"   Number of shells: {basis.nshells()}")
print(f"   Number of basis functions: {basis.nbf()}")

# Set basis tolerance
basis.set_basis_tolerance(1e-10)

# 3. Generate molecular grid
print("\n3. Generating molecular grid...")
grid = gxc.compute_grid(
    mol, basis,
    pruning_scheme="robust",
    batch_size=512,
    radial_quad="mura_knowles",
    grid_size="ultrafine",
    exec_space="host"
)

print(f"   Grid generated with {grid.npts} points")
print(f"   Points shape: {grid.points.shape}")
print(f"   Weights shape: {grid.weights.shape}")
print(f"   Total weight (approx volume): {np.sum(grid.weights):.2f} a.u.³")

# 4. Create a simple density matrix
# For demonstration, use a random symmetric matrix
print("\n4. Creating density matrix...")
nbf = basis.nbf()
np.random.seed(42)
P_random = np.random.randn(nbf, nbf)
P = (P_random + P_random.T) / 2  # Make symmetric
P = P * 0.1  # Scale down

# Normalize to approximately correct electron count
n_electrons = 10  # H2O has 10 electrons
trace_P_S = np.trace(P)  # Assuming S ≈ I for simplicity
if abs(trace_P_S) > 1e-10:
    P = P * (n_electrons / (2 * trace_P_S))  # Factor of 2 for RKS

print(f"   Density matrix shape: {P.shape}")
print(f"   Density matrix trace: {np.trace(P):.4f}")

# 5. Evaluate meta-GGA variables (would require full implementation)
print("\n5. Meta-GGA variable evaluation...")
print("   Note: Full vvar evaluation requires complete integrator binding")
print("   This is a placeholder showing the intended API:")
print()
print("   # This would evaluate rho, grad, gamma, tau on the grid:")
print("   # vvars = gxc.eval_mgga_vvars(")
print("   #     mol, basis, grid, P,")
print("   #     ks_scheme='RKS',")
print("   #     need_lapl=False")
print("   # )")
print("   # print(f'   rho shape: {vvars.rho.shape}')")
print("   # print(f'   grad shape: {vvars.grad.shape}')")
print("   # print(f'   gamma shape: {vvars.gamma.shape}')")
print("   # print(f'   tau shape: {vvars.tau.shape}')")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)

# Optional: Test with PyTorch if available
print("\nChecking PyTorch support...")
try:
    import torch
    print("   PyTorch is available!")
    print(f"   PyTorch version: {torch.__version__}")
    
    # Convert to torch tensors
    coords_torch = torch.from_numpy(coords)
    P_torch = torch.from_numpy(P)
    
    print(f"   Converted coords to torch: {coords_torch.shape}, dtype={coords_torch.dtype}")
    print(f"   Converted P to torch: {P_torch.shape}, dtype={P_torch.dtype}")
    
    # Could use grid.to_torch() if grid were a GridData object
    
except ImportError:
    print("   PyTorch not available (optional)")

print()
