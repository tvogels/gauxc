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
    P: Union[np.ndarray, tuple, list],
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
    P : numpy.ndarray, torch.Tensor, or tuple/list of arrays
        Density matrix or matrices:
        - RKS: Single array (nbf, nbf) - symmetric density matrix
        - UKS: Tuple/list of two arrays [(nbf, nbf), (nbf, nbf)] 
               for spin-up and spin-down density matrices [Ps, Pz]
        Can be CPU or GPU tensors; will be handled appropriately
    ks_scheme : str
        Kohn-Sham scheme: 'RKS' or 'UKS' (default: 'RKS')
        - 'RKS': Restricted (single density matrix)
        - 'UKS': Unrestricted (two density matrices for spin-up/down)
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
        For UKS: rho contains total density (alpha + beta)
        
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
    
    Device Consistency:
    -------------------
    For UKS, all density matrices must be on the same device.
    An error will be raised if matrices are on different devices.
    """
    # Normalize KS scheme
    ks_scheme = ks_scheme.upper()
    
    # Handle UKS input (tuple/list of density matrices)
    is_uks = False
    if isinstance(P, (tuple, list)):
        if len(P) != 2:
            raise ValueError(
                f"UKS requires exactly 2 density matrices (Ps, Pz), got {len(P)}"
            )
        is_uks = True
        Ps, Pz = P
        
        # Check device consistency
        if hasattr(Ps, 'device') and hasattr(Pz, 'device'):
            if Ps.device != Pz.device:
                raise ValueError(
                    f"Density matrices must be on the same device. "
                    f"Got Ps on {Ps.device} and Pz on {Pz.device}"
                )
        
        # Auto-detect UKS if not specified
        if ks_scheme == "RKS":
            import warnings
            warnings.warn(
                "Detected 2 density matrices but ks_scheme='RKS'. "
                "Auto-switching to ks_scheme='UKS'",
                UserWarning
            )
            ks_scheme = "UKS"
    else:
        # Single matrix - must be RKS
        if ks_scheme == "UKS":
            raise ValueError(
                "UKS requires 2 density matrices [Ps, Pz]. "
                "Pass as tuple: P=(Ps, Pz)"
            )
        Ps = P
        Pz = None
    
    # Detect if input is a torch tensor and its device
    input_device = None
    is_torch = False
    
    def process_tensor(tensor, name="P"):
        """Process a single tensor, handling device conversion."""
        nonlocal input_device, is_torch
        
        if hasattr(tensor, 'device'):  # torch tensor
            is_torch = True
            try:
                import torch
                curr_device = tensor.device
                
                # Set input_device from first tensor
                if input_device is None:
                    input_device = curr_device
                
                # Automatic exec_space detection based on tensor device
                if exec_space == "host" or isinstance(exec_space, ExecutionSpace) and exec_space == ExecutionSpace.Host:
                    if tensor.is_cuda or (hasattr(tensor, 'is_hip') and tensor.is_hip):
                        # GPU tensor but host execution - need to copy to CPU
                        return tensor.cpu().numpy()
                    else:
                        return tensor.numpy()
                else:  # Device execution
                    # Check if tensor is on GPU
                    if not (tensor.is_cuda or (hasattr(tensor, 'is_hip') and tensor.is_hip)):
                        raise ValueError(
                            f"exec_space='device' requires GPU tensor input. "
                            f"Please move {name} to GPU first with tensor.cuda() or tensor.to('cuda')"
                        )
                    # For device execution, we still need CPU copy for current implementation
                    # TODO: Add native GPU tensor support when GauXC supports device pointers directly
                    return tensor.cpu().numpy()
            except ImportError:
                pass
        elif hasattr(tensor, 'cpu'):  # torch tensor (older API check)
            is_torch = True
            return tensor.cpu().numpy()
        
        return np.asarray(tensor, dtype=np.float64)
    
    # Process density matrices
    Ps_np = process_tensor(Ps, "Ps" if is_uks else "P")
    Pz_np = process_tensor(Pz, "Pz") if Pz is not None else None
    
    # Validate density matrices
    nbf = basis.nbf()
    if Ps_np.shape != (nbf, nbf):
        name = "Ps" if is_uks else "P"
        raise ValueError(f"Density matrix {name} shape {Ps_np.shape} doesn't match basis size {nbf}x{nbf}")
    
    if is_uks and Pz_np.shape != (nbf, nbf):
        raise ValueError(f"Density matrix Pz shape {Pz_np.shape} doesn't match basis size {nbf}x{nbf}")
    
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
    if is_uks:
        result = eval_mgga_vvars_impl_uks(mol, basis, molgrid, Ps_np, Pz_np, 
                                           need_lapl, exec_space)
    else:
        result = eval_mgga_vvars_impl(mol, basis, molgrid, Ps_np, need_lapl, exec_space)
    
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
