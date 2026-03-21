#!/usr/bin/env python3
"""
Run GauXC scaling benchmark across molecules from Azure blob storage.

Usage:
    python run_scaling_benchmark.py --build-dir <path> [--model <path.fun>]
        [--device] [--iters N] [--warmup N] [--output results.json]
        [--molecules 1,2,3 | --molecules all]

Requires the `livdft` conda environment with ai4s_filesystems installed.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BLOB_BASE = "ai4science0eastus://livdft-data/cost-scaling-benchmark"
MOLECULE_COUNT = 23


def download_molecule(idx: int, cache_dir: Path) -> Path:
    """Download a molecule HDF5 file from Azure blob storage, with caching."""
    filename = f"{idx:02d}-skala-1.0.h5"
    local_path = cache_dir / filename
    if local_path.exists():
        return local_path

    from ai4s_filesystems import utils as fsutils
    from livdft.common.filesystems import register_fsspec_filesystems
    register_fsspec_filesystems()

    url = f"{BLOB_BASE}/{filename}"
    fs, path = fsutils.split_fs_and_path(url)
    fs.get(path, str(local_path))
    return local_path


def run_benchmark(
    benchmark_exe: str,
    input_file: str,
    models: list[tuple[str, str]] | None = None,
    device: bool = False,
    iters: int = 5,
    warmup: int = 3,
    omp_threads: int | None = None,
) -> dict | None:
    """Run the C++ benchmark and return parsed JSON, or None on failure."""
    nvidia_base = os.path.expanduser(
        "~/.local/lib/python3.11/site-packages/nvidia"
    )
    nvidia_libs = ":".join(
        f"{nvidia_base}/{d}/lib"
        for d in [
            "cuda_runtime", "cuda_cupti", "cublas", "cuda_nvrtc", "cudnn",
            "cufft", "cusolver", "cusparse", "cusparselt", "nvjitlink",
            "nvtx", "cufile", "curand",
        ]
    )
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{nvidia_libs}:/usr/local/cuda-13.1/lib64:" + env.get("LD_LIBRARY_PATH", "")
    if omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_threads)

    cmd = [benchmark_exe, input_file, "--json", "--iters", str(iters), "--warmup", str(warmup)]
    for name, path in (models or []):
        cmd += ["--model", f"{name}:{path}"]
    if device:
        cmd += ["--device", "--gpu-mem-frac", "0.1"]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, env=env
        )
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}): {result.stderr[-200:]}", file=sys.stderr)
            return None
        # Parse JSON from stdout (skip any non-JSON lines on stderr)
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="GauXC scaling benchmark")
    parser.add_argument("--build-dir", required=True, help="Path to GauXC build directory")
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "PATH"),
                        help="Model name and path (can be repeated)")
    parser.add_argument("--device", action="store_true", help="Use GPU execution")
    parser.add_argument("--iters", type=int, default=5, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--omp-threads", type=int, default=None, help="OMP_NUM_THREADS for CPU runs")
    parser.add_argument("--output", default="scaling_results.json", help="Output JSON file")
    parser.add_argument("--molecules", default="all",
                        help="Comma-separated molecule indices (1-23) or 'all'")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to cache downloaded HDF5 files")
    args = parser.parse_args()

    benchmark_exe = str(Path(args.build_dir) / "tests" / "benchmark")
    if not os.path.isfile(benchmark_exe):
        print(f"Error: benchmark executable not found at {benchmark_exe}", file=sys.stderr)
        sys.exit(1)

    if args.molecules == "all":
        mol_indices = list(range(1, MOLECULE_COUNT + 1))
    else:
        mol_indices = [int(x) for x in args.molecules.split(",")]

    # Setup cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="gauxc_bench_"))

    models = [(name, path) for name, path in (args.model or [])]

    print(f"Benchmark: {len(mol_indices)} molecules, "
          f"{'GPU' if args.device else 'CPU'}, "
          f"{args.iters} iters, {args.warmup} warmup"
          + (f", OMP_NUM_THREADS={args.omp_threads}" if args.omp_threads else "")
          + (f", models: {[n for n,_ in models]}" if models else ""))
    print(f"Cache dir: {cache_dir}")
    print()

    all_results = []
    for idx in mol_indices:
        print(f"[{idx:02d}/{MOLECULE_COUNT}] Downloading...", end=" ", flush=True)
        try:
            local_path = download_molecule(idx, cache_dir)
        except Exception as e:
            print(f"download failed: {e}", file=sys.stderr)
            continue

        # Run built-in functionals (no models) — always succeeds
        base_result = run_benchmark(
            benchmark_exe, str(local_path),
            models=[], device=args.device,
            iters=args.iters, warmup=args.warmup,
            omp_threads=args.omp_threads,
        )
        if not base_result:
            print("failed (base)")
            continue

        # Run each model separately to isolate OOM crashes
        for model_name, model_path in models:
            model_result = run_benchmark(
                benchmark_exe, str(local_path),
                models=[(model_name, model_path)], device=args.device,
                iters=args.iters, warmup=args.warmup,
                omp_threads=args.omp_threads,
            )
            if model_result:
                # Merge model results into base
                for r in model_result["results"]:
                    if r["type"] == "neural":
                        base_result["results"].append(r)

        base_result["molecule_index"] = idx
        all_results.append(base_result)
        nbf = base_result.get("nbf", "?")
        natoms = base_result.get("natoms", "?")
        n_results = len(base_result.get("results", []))
        print(f"done ({natoms} atoms, {nbf} basis fns, {n_results} functionals)")

    # Write combined results
    output = {
        "exec_space": "Device" if args.device else "Host",
        "models": {name: path for name, path in models},
        "omp_threads": args.omp_threads,
        "iters": args.iters,
        "warmup": args.warmup,
        "molecules": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {args.output} ({len(all_results)} molecules)")


if __name__ == "__main__":
    main()
