"""
Meta-GGA variable evaluation utilities.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from .core import (
    Molecule, BasisSet, MolGrid, ExecutionSpace,
    eval_mgga_vvars_impl
)
from .grid_utils import GridData


@dataclass  
class MGGAVariables:
    """Container for meta-GGA variables."""
    rho: np.ndarray      # (npts,) electron density
    grad: np.ndarray     # (npts, 3) density gradient
    gamma: np.ndarray    # (npts,) contracted gradient
    tau: np.ndarray      # (npts,) kinetic energy density
    lapl: Optional[np.ndarray] = None  # (npts,) Laplacian (optional)
    
    def to_torch(self):
        """Convert to PyTorch tensors (if available)."""
        try:
            import torch
            return MGGAVariables(
                rho=torch.from_numpy(self.rho),
                grad=torch.from_numpy(self.grad),
                gamma=torch.from_numpy(self.gamma),
                tau=torch.from_numpy(self.tau),
                lapl=torch.from_numpy(self.lapl) if self.lapl is not None else None
            )
        except ImportError:
            raise ImportError("PyTorch is not installed")


def eval_mgga_vvars(
    mol: Molecule,
    basis: BasisSet,
    grid: Union[GridData, MolGrid],
    P: np.ndarray,
    ks_scheme: str = "RKS",
    need_lapl: bool = False,
    exec_space: Union[str, ExecutionSpace] = "host",
    return_torch: bool = False
) -> MGGAVariables:
    """
    Evaluate meta-GGA variables on a molecular grid.
    
    This function computes density and related quantities needed for meta-GGA
    functionals by evaluating basis functions on the grid and contracting with
    the density matrix.
    
    Parameters:
    -----------
    mol : Molecule
        The molecule
    basis : BasisSet
        The basis set
    grid : GridData or MolGrid
        The molecular grid (from compute_grid or create_molgrid)
    P : numpy.ndarray or torch.Tensor
        Density matrix (nbf x nbf, symmetric)
    ks_scheme : str
        Kohn-Sham scheme: 'RKS', 'UKS', or 'GKS' (default: 'RKS')
        Note: Currently only RKS is implemented
    need_lapl : bool
        Whether to compute Laplacian (default: False)
    exec_space : str or ExecutionSpace
        Execution space: 'host' or 'device' (default: 'host')
    return_torch : bool
        Return PyTorch tensors instead of numpy arrays (default: False)
        
    Returns:
    --------
    MGGAVariables
        Computed meta-GGA variables (rho, grad, gamma, tau, lapl)
        
    Notes:
    ------
    - rho: electron density ρ(r) = Σᵢⱼ Pᵢⱼ φᵢ(r) φⱼ(r)
    - grad: density gradient ∇ρ(r) = (∂ρ/∂x, ∂ρ/∂y, ∂ρ/∂z)
    - gamma: contracted gradient γ(r) = |∇ρ(r)|² = (∂ρ/∂x)² + (∂ρ/∂y)² + (∂ρ/∂z)²
    - tau: kinetic energy density τ(r) = ½ Σᵢⱼ Pᵢⱼ (∇φᵢ(r))·(∇φⱼ(r))
    - lapl: Laplacian ∇²ρ(r) = ∂²ρ/∂x² + ∂²ρ/∂y² + ∂²ρ/∂z² (optional)
    """
    # Convert torch tensors to numpy if needed
    if hasattr(P, 'cpu'):  # torch tensor
        P = P.cpu().numpy()
    
    P = np.asarray(P, dtype=np.float64)
    
    # Validate density matrix
    nbf = basis.nbf()
    if P.shape != (nbf, nbf):
        raise ValueError(f"Density matrix shape {P.shape} doesn't match basis size {nbf}x{nbf}")
    
    # Convert exec_space string to enum if needed
    if isinstance(exec_space, str):
        exec_map = {
            'host': ExecutionSpace.Host,
        }
        try:
            exec_map['device'] = ExecutionSpace.Device
        except AttributeError:
            pass
        exec_space = exec_map[exec_space.lower()]
    
    # Check KS scheme (only RKS implemented currently)
    if ks_scheme.upper() != "RKS":
        raise NotImplementedError(
            f"KS scheme '{ks_scheme}' not yet implemented. "
            "Currently only 'RKS' (restricted) is supported."
        )
    
    # Extract MolGrid if GridData was passed
    if isinstance(grid, GridData):
        if grid.molgrid is None:
            raise ValueError(
                "GridData object does not contain MolGrid. "
                "Please regenerate the grid or pass MolGrid directly."
            )
        molgrid = grid.molgrid
    else:
        molgrid = grid
    
    # Call C++ implementation
    result = eval_mgga_vvars_impl(mol, basis, molgrid, P, need_lapl, exec_space)
    
    # Create MGGAVariables object
    vvars = MGGAVariables(
        rho=result['rho'],
        grad=result['grad'],
        gamma=result['gamma'],
        tau=result['tau'],
        lapl=result.get('lapl', None)
    )
    
    # Convert to torch if requested
    if return_torch:
        vvars = vvars.to_torch()
    
    return vvars
