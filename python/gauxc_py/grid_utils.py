"""
Grid generation utilities.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from .core import (
    Molecule, BasisSet, MolGrid,
    create_molgrid, extract_grid_data,
    PruningScheme, RadialQuad, AtomicGridSizeDefault,
    BatchSize, ExecutionSpace
)


@dataclass
class GridData:
    """Container for molecular grid data."""
    points: np.ndarray  # (npts, 3) grid point coordinates
    weights: np.ndarray  # (npts,) quadrature weights
    npts: int  # Total number of points
    
    def to_torch(self):
        """Convert to PyTorch tensors (if available)."""
        try:
            import torch
            return GridData(
                points=torch.from_numpy(self.points),
                weights=torch.from_numpy(self.weights),
                npts=self.npts
            )
        except ImportError:
            raise ImportError("PyTorch is not installed")


def compute_grid(
    mol: Molecule,
    basis: BasisSet,
    pruning_scheme: Union[str, PruningScheme] = "robust",
    batch_size: int = 512,
    radial_quad: Union[str, RadialQuad] = "mura_knowles",
    grid_size: Union[str, AtomicGridSizeDefault] = "ultrafine",
    exec_space: Union[str, ExecutionSpace] = "host",
    return_torch: bool = False
) -> GridData:
    """
    Generate a molecular integration grid.
    
    Parameters:
    -----------
    mol : Molecule
        The molecule
    basis : BasisSet
        The basis set
    pruning_scheme : str or PruningScheme
        Grid pruning scheme: 'unpruned', 'robust', or 'treutler'
    batch_size : int
        Batch size for grid generation (default: 512)
    radial_quad : str or RadialQuad
        Radial quadrature: 'becke', 'mura_knowles', 'treutler_ahlrichs', 
        or 'murray_handy_laming' (default: 'mura_knowles')
    grid_size : str or AtomicGridSizeDefault
        Grid quality: 'fine', 'ultrafine', 'superfine', 'gm3', or 'gm5'
        (default: 'ultrafine')
    exec_space : str or ExecutionSpace
        Execution space: 'host' or 'device' (default: 'host')
    return_torch : bool
        Return PyTorch tensors instead of numpy arrays (default: False)
        
    Returns:
    --------
    GridData
        Grid points, weights, and metadata
    """
    # Convert string arguments to enums
    if isinstance(pruning_scheme, str):
        pruning_map = {
            'unpruned': PruningScheme.Unpruned,
            'robust': PruningScheme.Robust,
            'treutler': PruningScheme.Treutler,
        }
        pruning_scheme = pruning_map[pruning_scheme.lower()]
    
    if isinstance(radial_quad, str):
        radial_map = {
            'becke': RadialQuad.Becke,
            'mura_knowles': RadialQuad.MuraKnowles,
            'treutler_ahlrichs': RadialQuad.TreutlerAhlrichs,
            'murray_handy_laming': RadialQuad.MurrayHandyLaming,
            'mk': RadialQuad.MuraKnowles,
            'ta': RadialQuad.TreutlerAhlrichs,
            'mhl': RadialQuad.MurrayHandyLaming,
        }
        radial_quad = radial_map[radial_quad.lower()]
    
    if isinstance(grid_size, str):
        size_map = {
            'fine': AtomicGridSizeDefault.FineGrid,
            'ultrafine': AtomicGridSizeDefault.UltraFineGrid,
            'superfine': AtomicGridSizeDefault.SuperFineGrid,
            'gm3': AtomicGridSizeDefault.GM3,
            'gm5': AtomicGridSizeDefault.GM5,
        }
        grid_size = size_map[grid_size.lower()]
    
    if isinstance(exec_space, str):
        exec_map = {
            'host': ExecutionSpace.Host,
        }
        # Add device if available
        try:
            exec_map['device'] = ExecutionSpace.Device
        except AttributeError:
            pass
        exec_space = exec_map[exec_space.lower()]
    
    # Create molecular grid
    molgrid = create_molgrid(mol, pruning_scheme, BatchSize(batch_size),
                             radial_quad, grid_size)
    
    # Extract grid data
    grid_dict = extract_grid_data(mol, molgrid, basis, exec_space)
    
    # Create GridData object
    grid_data = GridData(
        points=grid_dict['points'],
        weights=grid_dict['weights'],
        npts=grid_dict['npts']
    )
    
    # Convert to torch if requested
    if return_torch:
        grid_data = grid_data.to_torch()
    
    return grid_data
