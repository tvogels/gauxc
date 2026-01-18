"""
Core Python wrappers around C++ bindings.
"""

try:
    from ._gauxc_core import *
except ImportError as e:
    raise ImportError(
        "Failed to import GauXC core module. "
        "Make sure the package is properly installed. "
        f"Error: {e}"
    )

import numpy as np

# Re-export for convenience
__all__ = [
    'Molecule',
    'Atom',
    'AtomicNumber',
    'BasisSet',
    'Shell',
    'PrimSize',
    'AngularMomentum',
    'SphericalType',
    'BatchSize',
    'RadialSize',
    'AngularSize',
    'RadialScale',
    'RadialQuad',
    'PruningScheme',
    'AtomicGridSizeDefault',
    'ExecutionSpace',
    'Grid',
    'MolGrid',
    'LoadBalancer',
    'IntegratorSettingsXC',
    'IntegratorSettingsEXX',
    'create_molecule',
    'get_molecule_coords',
    'create_simple_shell',
    'create_molgrid',
    'extract_grid_data',
    'eval_mgga_vvars',
    'supports_torch',
]


def molecule_from_arrays(atomic_numbers, coords):
    """
    Create a Molecule from numpy arrays or torch tensors.
    
    Parameters:
    -----------
    atomic_numbers : array-like, shape (natoms,)
        Atomic numbers
    coords : array-like, shape (natoms, 3)
        Atomic coordinates in Angstroms
        
    Returns:
    --------
    Molecule
    """
    # Convert to numpy if needed
    if hasattr(atomic_numbers, 'cpu'):  # torch tensor
        atomic_numbers = atomic_numbers.cpu().numpy()
    if hasattr(coords, 'cpu'):  # torch tensor
        coords = coords.cpu().numpy()
    
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int32)
    coords = np.asarray(coords, dtype=np.float64)
    
    return create_molecule(atomic_numbers, coords)
