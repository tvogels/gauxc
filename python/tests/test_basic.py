"""
Basic tests for GauXC Python bindings.
"""

import pytest
import numpy as np

try:
    import gauxc_py as gxc
    GAUXC_AVAILABLE = True
except ImportError:
    GAUXC_AVAILABLE = False


@pytest.mark.skipif(not GAUXC_AVAILABLE, reason="GauXC not installed")
class TestMolecule:
    """Test Molecule creation and manipulation."""
    
    def test_create_empty_molecule(self):
        mol = gxc.Molecule()
        assert mol.natoms() == 0
    
    def test_create_molecule_from_arrays(self):
        z = np.array([1, 1], dtype=np.int32)
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=np.float64)
        mol = gxc.molecule_from_arrays(z, coords)
        assert mol.natoms() == 2
        assert mol[0].Z.get() == 1
    
    def test_molecule_add_atoms(self):
        mol = gxc.Molecule()
        atom = gxc.Atom(gxc.AtomicNumber(8), 0.0, 0.0, 0.0)
        mol.append(atom)
        assert mol.natoms() == 1
        assert mol[0].Z.get() == 8


@pytest.mark.skipif(not GAUXC_AVAILABLE, reason="GauXC not installed")
class TestBasisSet:
    """Test BasisSet creation."""
    
    def test_create_empty_basis(self):
        basis = gxc.BasisSet()
        assert basis.nshells() == 0
        assert basis.nbf() == 0
    
    def test_create_simple_shell(self):
        alphas = np.array([3.4253, 0.6239, 0.1689], dtype=np.float64)
        coeffs = np.array([0.1543, 0.5353, 0.4446], dtype=np.float64)
        
        shell = gxc.create_simple_shell(1, 0, alphas, coeffs, 0.0, 0.0, 0.0)
        assert shell.nprim() == 3
        assert shell.l() == 0
    
    def test_basis_with_shells(self):
        basis = gxc.BasisSet()
        
        alphas = np.array([3.4253, 0.6239, 0.1689], dtype=np.float64)
        coeffs = np.array([0.1543, 0.5353, 0.4446], dtype=np.float64)
        
        shell1 = gxc.create_simple_shell(1, 0, alphas, coeffs, 0.0, 0.0, 0.0)
        shell2 = gxc.create_simple_shell(1, 0, alphas, coeffs, 0.0, 0.0, 1.4)
        
        basis.append(shell1)
        basis.append(shell2)
        
        assert basis.nshells() == 2
        # For s shells (l=0), size = 1
        assert basis.nbf() == 2


@pytest.mark.skipif(not GAUXC_AVAILABLE, reason="GauXC not installed")
class TestGrid:
    """Test grid generation."""
    
    def test_grid_enums(self):
        # Test that enums are accessible
        assert hasattr(gxc, 'PruningScheme')
        assert hasattr(gxc, 'RadialQuad')
        assert hasattr(gxc, 'AtomicGridSizeDefault')
        assert hasattr(gxc, 'ExecutionSpace')
    
    def test_compute_grid_h2(self):
        # Create H2 molecule
        z = np.array([1, 1], dtype=np.int32)
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=np.float64)
        mol = gxc.molecule_from_arrays(z, coords)
        
        # Create minimal basis
        basis = gxc.BasisSet()
        alphas = np.array([3.4253, 0.6239, 0.1689], dtype=np.float64)
        coeffs = np.array([0.1543, 0.5353, 0.4446], dtype=np.float64)
        
        for i in range(2):
            shell = gxc.create_simple_shell(
                1, 0, alphas, coeffs,
                coords[i, 0], coords[i, 1], coords[i, 2]
            )
            basis.append(shell)
        
        # Generate grid
        grid = gxc.compute_grid(
            mol, basis,
            pruning_scheme="robust",
            batch_size=512,
            grid_size="fine",
            exec_space="host"
        )
        
        assert grid.npts > 0
        assert grid.points.shape[0] == grid.npts
        assert grid.points.shape[1] == 3
        assert grid.weights.shape[0] == grid.npts
        assert np.all(grid.weights > 0)  # All weights should be positive


@pytest.mark.skipif(not GAUXC_AVAILABLE, reason="GauXC not installed")
class TestTorchIntegration:
    """Test PyTorch integration (optional)."""
    
    def test_torch_available(self):
        # Just check if the function exists
        assert hasattr(gxc, 'supports_torch')
        supports = gxc.supports_torch()
        # Result depends on whether torch is installed
        assert isinstance(supports, bool)
    
    @pytest.mark.skipif(not GAUXC_AVAILABLE, reason="GauXC not installed")
    def test_tensor_conversion(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        # Create molecule from torch tensors
        z_torch = torch.tensor([1, 1], dtype=torch.int32)
        coords_torch = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], 
                                     dtype=torch.float64)
        
        mol = gxc.molecule_from_arrays(z_torch, coords_torch)
        assert mol.natoms() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
