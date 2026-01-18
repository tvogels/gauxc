"""
Meta-GGA variable evaluation utilities.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from .core import (
    Molecule, BasisSet, MolGrid, ExecutionSpace,
    eval_mgga_vvars_stub
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
    grid: GridData,
    P: np.ndarray,
    ks_scheme: str = "RKS",
    need_lapl: bool = False,
    exec_space: Union[str, ExecutionSpace] = "host",
    return_torch: bool = False
) -> MGGAVariables:
    """
    Evaluate meta-GGA variables on a molecular grid.
    
    NOTE: This is currently a stub implementation that returns zero arrays.
    Full implementation requires deeper integration with GauXC collocation
    and density evaluation kernels.
    
    Parameters:
    -----------
    mol : Molecule
        The molecule
    basis : BasisSet
        The basis set
    grid : GridData
        The molecular grid (from compute_grid)
    P : numpy.ndarray or torch.Tensor
        Density matrix (nbf x nbf)
    ks_scheme : str
        Kohn-Sham scheme: 'RKS', 'UKS', or 'GKS' (default: 'RKS')
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
    - rho: electron density ρ(r)
    - grad: density gradient ∇ρ(r) = (∂ρ/∂x, ∂ρ/∂y, ∂ρ/∂z)
    - gamma: contracted gradient γ(r) = |∇ρ(r)|²
    - tau: kinetic energy density τ(r) = ½ Σᵢ |∇ψᵢ(r)|²
    - lapl: Laplacian ∇²ρ(r) (optional)
    """
    # Convert torch tensors to numpy if needed
    if hasattr(P, 'cpu'):  # torch tensor
        P = P.cpu().numpy()
    
    P = np.asarray(P, dtype=np.float64)
    
    # Validate density matrix
    nbf = basis.nbf()
    if P.shape != (nbf, nbf):
        raise ValueError(f"Density matrix shape {P.shape} doesn't match basis size {nbf}x{nbf}")
    
    # Call C++ stub (returns zeros for now)
    result = eval_mgga_vvars_stub(grid.points, P)
    
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
