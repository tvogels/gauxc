# OneDFT Benchmarks

Tools for benchmarking and validating GauXC's OneDFT (neural functional) integration.

## Tools

### `benchmark` (C++)

Times multiple XC functionals on a single molecule. Runs built-in functionals
(SVWN5, revPBE, R2SCAN, B3LYP, M06-2X) and any number of OneDFT neural models.

```bash
# Basic usage (CPU)
./build/tests/benchmark molecule.h5

# GPU with a neural model
./build/tests/benchmark molecule.h5 --device --model "Skala 1.1:/path/to/model.fun"

# JSON output for scripting
./build/tests/benchmark molecule.h5 --json --model "Skala 1.1:model.fun"

# MPI parallel
mpirun -np 2 ./build/tests/benchmark molecule.h5 --device
```

**Flags:**
| Flag | Description |
|------|-------------|
| `--device` | Use GPU (CUDA) execution |
| `--model name:path` | Add a neural functional (repeatable) |
| `--json` | Machine-readable JSON output |
| `--iters N` | Timed iterations (default: 5) |
| `--warmup N` | Warmup iterations (default: 1) |
| `--gpu-mem-frac F` | Fraction of GPU memory for GauXC (default: 0.9; use ~0.1 with OneDFT to leave room for PyTorch) |

**Notes:**
- Hybrid functionals (B3LYP, M06-2X) are skipped for molecules with >2000 basis functions to avoid GPU OOM from sn-K exact exchange.
- GPU sn-K only supports angular momentum L≤2; the benchmark uses CPU fallback for higher L.
- Each OneDFT model is run in a try/catch — if a model OOMs or crashes, the benchmark continues with the next.

### `onedft_driver` (C++)

Minimal driver for testing a single OneDFT model. Useful for quick validation
and debugging.

```bash
# CPU
./build/tests/onedft_driver molecule.h5 /path/to/model.fun

# GPU
./build/tests/onedft_driver molecule.h5 /path/to/model.fun --gpu
```

Prints EXC, VXC norms, and timing. Runs two iterations and checks consistency.

### `run_scaling_benchmark.py` (Python)

Orchestrator that downloads 23 molecules (increasing size, def2-TZVP basis) from
Azure blob storage and runs `benchmark` on each. Each model runs as a separate
subprocess for OOM crash isolation.

```bash
python tests/onedft_benchmarks/run_scaling_benchmark.py \
    --build-dir build-cuda \
    --device \
    --model "Skala 1.1:/path/to/model.fun" \
    --output results.json
```

**Requirements:** `livdft` package (which provides `ai4s_filesystems` and Azure blob access).

## HDF5 Input Format

All tools expect an HDF5 file with these datasets:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `MOLECULE/COORDS` | (natoms, 3) | Atomic coordinates in Bohr |
| `MOLECULE/ATOMS` | (natoms,) | Atomic numbers |
| `BASIS/...` | — | Basis set specification |
| `DENSITY_SCALAR` | (nbf, nbf) | Alpha+beta density matrix (P_s) |
| `DENSITY_Z` | (nbf, nbf) | Alpha−beta density matrix (P_z) |

These files can be generated from PySCF using `skala.gauxc.export.write_gauxc_h5_from_pyscf`.

## JSON Output Schema

When `--json` is passed to `benchmark`, it writes a JSON object:

```json
{
  "input_file": "molecule.h5",
  "natoms": 3,
  "nbf": 24,
  "nshells": 11,
  "grid_pts": 46062,
  "mpi_ranks": 1,
  "exec_space": "Device",
  "warmup_iters": 1,
  "timed_iters": 5,
  "results": [
    {
      "name": "SVWN5",
      "type": "LDA",
      "description": "XC energy + potential (eval_exc_vxc)",
      "warmup_s": 0.015,
      "mean_s": 0.012,
      "min_s": 0.011,
      "std_s": 0.001,
      "times_s": [0.012, 0.011, 0.013, 0.012, 0.011]
    }
  ]
}
```

The `run_scaling_benchmark.py` wraps per-molecule results into:
```json
{
  "molecules": [ { ...per-molecule result... } ]
}
```
