"""
GauXC Python Bindings

A Python interface to the GauXC library for GPU-accelerated 
exchange-correlation calculations with PyTorch integration.
"""

from .core import *
from .grid_utils import compute_grid, GridData
from .vvar_eval import eval_mgga_vvars

__version__ = "1.0.0"

__all__ = [
    # Core classes
    'Molecule',
    'BasisSet', 
    'Shell',
    'AtomicNumber',
    'Atom',
    
    # Grid utilities
    'compute_grid',
    'GridData',
    'PruningScheme',
    'RadialQuad',
    'AtomicGridSizeDefault',
    'ExecutionSpace',
    
    # Variable evaluation
    'eval_mgga_vvars',
    
    # Version
    '__version__',
]
