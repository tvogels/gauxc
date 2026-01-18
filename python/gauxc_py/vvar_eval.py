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
    
    def to_torch(self, device=None):
        """
        Convert to PyTorch tensors (if available).
        
        Parameters:
        -----------
        device : str or torch.device, optional
            Target device for tensors. If None, keeps on current device.
        """
        try:
            import torch
            
            def convert_array(arr, device):
                if arr is None:
                    return None
                tensor = torch.from_numpy(arr)
                if device is not None:
                    tensor = tensor.to(device)
                return tensor
            
            return MGGAVariables(
                rho=convert_array(self.rho, device),
                grad=convert_array(self.grad, device),
                gamma=convert_array(self.gamma, device),
                tau=convert_array(self.tau, device),
                lapl=convert_array(self.lapl, device) if self.lapl is not None else None
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
    return_torch: bool = False,
    device: Optional[str] = None
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
        Can be CPU or GPU tensor; will be handled appropriately
    ks_scheme : str
        Kohn-Sham scheme: 'RKS', 'UKS', or 'GKS' (default: 'RKS')
        Note: Currently only RKS is implemented
    need_lapl : bool
        Whether to compute Laplacian (default: False)
    exec_space : str or ExecutionSpace
        Execution space: 'host' or 'device' (default: 'host')
    return_torch : bool
        Return PyTorch tensors instead of numpy arrays (default: False)
    device : str or torch.device, optional
        Target device for output when return_torch=True
        If None and input is GPU tensor, uses same device as input
        
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
    
    GPU Tensor Support:
    -------------------
    When exec_space='device' and CUDA/HIP is enabled:
    - GPU tensors can be passed directly and will be handled efficiently
    - Results can be returned as GPU tensors with return_torch=True
    - Zero-copy operations used when possible
    
    When exec_space='host' (default):
    - GPU tensors are automatically copied to CPU for computation
    - Results are returned on CPU
    """
    # Detect if input is a torch tensor and its device
    input_device = None
    is_torch = False
    
    if hasattr(P, 'device'):  # torch tensor
        is_torch = True
        try:
            import torch
            input_device = P.device
            
            # Automatic exec_space detection based on tensor device
            if exec_space == "host" or isinstance(exec_space, ExecutionSpace) and exec_space == ExecutionSpace.Host:
                if P.is_cuda or (hasattr(P, 'is_hip') and P.is_hip):
                    # GPU tensor but host execution - need to copy to CPU
                    P = P.cpu().numpy()
                else:
                    P = P.numpy()
            else:  # Device execution
                # Check if tensor is on GPU
                if not (P.is_cuda or (hasattr(P, 'is_hip') and P.is_hip)):
                    raise ValueError(
                        "exec_space='device' requires GPU tensor input. "
                        "Please move tensor to GPU first with tensor.cuda() or tensor.to('cuda')"
                    )
                # For device execution, we still need CPU copy for current implementation
                # TODO: Add native GPU tensor support when GauXC supports device pointers directly
                P = P.cpu().numpy()
        except ImportError:
            pass
    elif hasattr(P, 'cpu'):  # torch tensor (older API check)
        is_torch = True
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
            if exec_space.lower() == 'device':
                raise RuntimeError(
                    "Device execution requested but GauXC was not built with CUDA/HIP support. "
                    "Please rebuild with -DGAUXC_ENABLE_CUDA=ON or -DGAUXC_ENABLE_HIP=ON"
                )
        exec_space = exec_map.get(exec_space.lower(), ExecutionSpace.Host)
    
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
        # Determine target device
        if device is None and is_torch and input_device is not None:
            # Use same device as input
            device = input_device
        vvars = vvars.to_torch(device)
    
    return vvars
