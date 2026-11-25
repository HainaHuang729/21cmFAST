"""Phase 2 benchmark: 192^3 / 64^3 native C/CUDA Eulerian lightcone vs CPU and Phase 0."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

import numpy as np
from powerbox import get_power

import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton as manager,
)
from py21cmfast.lightconers import RectilinearLightconer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIELDS: tuple[str, ...] = (
    "brightness_temp",
    "neutral_fraction",
    "xray_ionised_fraction",
    "kinetic_temp_neutral",
    "spin_temperature",
    "cumulative_recombinations",
)

STAT_FIELDS: tuple[str, ...] = (
    "compiled",
    "enabled",
    "device_id",
    "sm_count",
    "free_mem_bytes",
    "total_mem_bytes",
    "mempool_alloc_calls",
    "mempool_free_calls",
    "mempool_bytes_allocated",
    "cufft_plan_hits",
    "cufft_plan_misses",
    "cufft_plan_evictions",
    "cufft_exec_r2c",
    "cufft_exec_c2r",
    "cufft_exec_c2c",
    "registry_hits",
    "registry_misses",
    "registry_invalidations",
    "registry_h2d_bytes",
    "registry_d2h_bytes",
    "ic_runs",
    "perturb_runs",
    "emissivity_runs",
    "radiation_setup_runs",
    "radiation_update_runs",
    "spin_temp_runs",
    "ionization_runs",
    "brightness_runs",
)


def make_benchmark_case(
    *,
    n_threads: int = 1,
) -> tuple[p21.InputParameters, RectilinearLightconer]:
    """Construct the Phase 0 B1 192^3 / 64^3 Eulerian benchmark inputs and lightconer.

    Parameters
    ----------
    n_threads : int, default 1
        OpenMP thread count passed to ``SimulationOptions.N_THREADS``.

    Returns
    -------
    tuple[InputParameters, RectilinearLightconer]
        Input parameter bundle and matching rectilinear lightconer.
    """
    from tests import produce_integration_test_data as official

    with patch.object(official, "SEED", 424242):
        inputs = official.get_all_options_struct(
            6,
            lc=True,
            DIM=192,
            HII_DIM=64,
            BOX_LEN=200,
            N_THREADS=n_threads,
            KEEP_3D_VELOCITIES=True,
            PERTURB_ON_HIGH_RES=True,
            SMOOTH_EVOLVED_DENSITY_FIELD=True,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
        )["inputs"]
    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift=6,
        max_redshift=15,
        quantities=FIELDS,
        resolution=inputs.simulation_options.cell_size,
    )
    return inputs, lightconer


def snapshot_cuda_stats() -> dict[str, int]:
    """Read the current ``P21CudaStats`` struct into a plain dictionary.

    Returns
    -------
    dict[str, int]
        Integer counters and byte metrics from the C/CUDA backend.
    """
    stats = ffi.new("P21CudaStats *")
    lib.p21_cuda_get_stats(stats)
    return {name: int(getattr(stats, name)) for name in STAT_FIELDS}


def compute_field_diff_stats(
    cpu_arr: np.ndarray, gpu_arr: np.ndarray
) -> dict[str, Any]:
    """Compute elementwise and RMS discrepancy metrics between CPU and GPU arrays.

    Parameters
    ----------
    cpu_arr : numpy.ndarray
        Reference CPU output array.
    gpu_arr : numpy.ndarray
        GPU output array with identical shape.

    Returns
    -------
    dict[str, Any]
        Summary statistics including ``max_abs``, ``p99_9_abs``, ``rel_rms``, and
        tolerance exceedance percentages.
    """
    x = np.asarray(cpu_arr, dtype=np.float64)
    y = np.asarray(gpu_arr, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: CPU {x.shape} != GPU {y.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Nonfinite values encountered during comparison")
    diff = np.abs(y - x)
    cpu_rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    diff_rms = float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0
    rel_rms = float(diff_rms / cpu_rms) if cpu_rms > 0.0 else float(diff_rms)
    nz = np.abs(x) > 1e-6
    rel = diff[nz] / np.abs(x[nz])
    bad_strict = diff > (1e-8 + 1e-4 * np.abs(x))
    bad_relaxed = diff > (1e-5 + 1e-4 * np.abs(x))
    return {
        "count": int(diff.size),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "p99_9_abs": float(np.percentile(diff, 99.9)) if diff.size else 0.0,
        "rel_rms": rel_rms,
        "max_rel_nonzero": float(rel.max()) if rel.size else 0.0,
        "p99_9_rel_nonzero": float(np.percentile(rel, 99.9)) if rel.size else 0.0,
        "strict_1e8_1e4_pct": float(100.0 * bad_strict.sum() / diff.size)
        if diff.size
        else 0.0,
        "relaxed_1e5_1e4_pct": float(100.0 * bad_relaxed.sum() / diff.size)
        if diff.size
        else 0.0,
    }


def run_single_pass(
    *,
    tag: str,
    use_cuda: bool,
    n_threads: int,
    work_dir: Path,
    cold_cuda: bool = False,
) -> tuple[dict[str, Any], p21.LightCone]:
    """Execute one full IC + 43-node Eulerian lightcone pass and record timings.

    Parameters
    ----------
    tag : str
        Identifier for the benchmark pass (for example ``'cuda_warm'``).
    use_cuda : bool
        Whether to enable ``USE_CUDA`` in configuration.
    n_threads : int
        Number of CPU threads to use in ``SimulationOptions.N_THREADS``.
    work_dir : pathlib.Path
        Scratch directory for cache isolation and temporary HDF5 output.
    cold_cuda : bool, default False
        If True, release C/CUDA caches and device pool before starting.

    Returns
    -------
    tuple[dict[str, Any], LightCone]
        Timing/stats record dictionary and computed ``LightCone`` object.
    """
    inputs, lightconer = make_benchmark_case(n_threads=n_threads)
    manager.free()
    if cold_cuda:
        lib.p21_cuda_cleanup()
    else:
        lib.p21_cuda_registry_clear()
    lib.p21_cuda_reset_stats()

    h5_path = work_dir / f"{tag}.h5"
    direc = work_dir / f"{tag}_cache"
    direc.mkdir(parents=True, exist_ok=True)

    with p21.config.use(USE_CUDA=use_cuda, direc=str(direc)):
        t0 = perf_counter()
        ic = p21.compute_initial_conditions(inputs=inputs, write=False, regenerate=True)
        t_ic = perf_counter()
        cone = p21.run_lightcone(
            lightconer=lightconer,
            inputs=inputs,
            initial_conditions=ic,
            write=False,
            regenerate=True,
        )
        t_compute_end = perf_counter()
        cone.save(h5_path, clobber=True)
        with h5_path.open("rb") as f:
            os.fsync(f.fileno())
        t_end = perf_counter()

    assert "cupy" not in sys.modules
    stats = snapshot_cuda_stats()
    record: dict[str, Any] = {
        "tag": tag,
        "use_cuda": use_cuda,
        "n_threads": n_threads,
        "nodes": len(inputs.node_redshifts),
        "ic_seconds": float(t_ic - t0),
        "lightcone_seconds": float(t_compute_end - t_ic),
        "compute_seconds": float(t_compute_end - t0),
        "total_with_save_seconds": float(t_end - t0),
        "cuda_stats": stats,
    }
    return record, cone


def _load_phase0_baseline(report_path: Path) -> dict[str, float]:
    """Load archived Phase 0 B1 metrics if available in the repository."""
    defaults = {
        "cpu_phase0_s": 65.77935682691168,
        "cupy_cufft_cold_s": 11.539228803943843,
        "cupy_cufft_warm_s": 10.33147444599308,
    }
    if not report_path.exists():
        return defaults
    data = json.loads(report_path.read_text())
    b1 = data.get("benchmark_1_eulerian_192", {})
    return {
        "cpu_phase0_s": float(b1.get("cpu_elapsed_seconds", defaults["cpu_phase0_s"])),
        "cupy_cufft_cold_s": float(
            b1.get("gpu_cold_seconds", defaults["cupy_cufft_cold_s"])
        ),
        "cupy_cufft_warm_s": float(
            b1.get("gpu_warm_seconds", defaults["cupy_cufft_warm_s"])
        ),
    }


def _compute_ps_comparison(
    cpu_cone: p21.LightCone, gpu_cone: p21.LightCone
) -> dict[str, Any]:
    """Compare 3D lightcone brightness-temperature power spectra between CPU and GPU."""
    res_cpu = get_power(
        cpu_cone.lightcones["brightness_temp"],
        boxlength=cpu_cone.lightcone_dimensions,
        bins_upto_boxlen=True,
    )
    res_gpu = get_power(
        gpu_cone.lightcones["brightness_temp"],
        boxlength=gpu_cone.lightcone_dimensions,
        bins_upto_boxlen=True,
    )
    k_cpu, p_cpu = np.asarray(res_cpu.bin_avg), np.asarray(res_cpu.power)
    k_gpu, p_gpu = np.asarray(res_gpu.bin_avg), np.asarray(res_gpu.power)
    assert np.allclose(k_cpu, k_gpu, rtol=1e-7, atol=0.0, equal_nan=True)
    mask = np.isfinite(p_cpu) & np.isfinite(p_gpu) & (np.abs(p_cpu) > 0.0)
    rel = np.abs(p_gpu[mask] - p_cpu[mask]) / np.abs(p_cpu[mask])
    return {
        "k_bins": int(mask.sum()),
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "median_rel": float(np.median(rel)) if rel.size else 0.0,
        "p99_9_rel": float(np.percentile(rel, 99.9)) if rel.size else 0.0,
    }


def main() -> None:
    """Run CPU (1T/2T/4T) and native CUDA (cold 2T, warm 1T/2T) Eulerian 192^3 benchmarks."""
    assert "cupy" not in sys.modules
    if lib.p21_cuda_is_compiled() != 1 or lib.p21_cuda_is_available() != 1:
        raise RuntimeError(
            "Native CUDA runtime must be compiled and available to run Phase 2 benchmark."
        )

    out_path = REPO_ROOT / "reports" / "phase2-eulerian-benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phase0_ref = _load_phase0_baseline(
        REPO_ROOT / "reports" / "phase0-cufft-baseline.json"
    )

    with tempfile.TemporaryDirectory(prefix="p21_phase2_bench_") as tmp_dir:
        work_dir = Path(tmp_dir)
        cuda_cold_t2_rec, _ = run_single_pass(
            tag="cuda_cold_t2",
            use_cuda=True,
            n_threads=2,
            work_dir=work_dir,
            cold_cuda=True,
        )
        cuda_warm_t2_rec, cuda_warm_t2_cone = run_single_pass(
            tag="cuda_warm_t2",
            use_cuda=True,
            n_threads=2,
            work_dir=work_dir,
            cold_cuda=False,
        )
        cuda_warm_t1_rec, _ = run_single_pass(
            tag="cuda_warm_t1",
            use_cuda=True,
            n_threads=1,
            work_dir=work_dir,
            cold_cuda=False,
        )
        cpu_t2_rec, cpu_t2_cone = run_single_pass(
            tag="cpu_t2",
            use_cuda=False,
            n_threads=2,
            work_dir=work_dir,
            cold_cuda=False,
        )
        cpu_t1_rec, _ = run_single_pass(
            tag="cpu_t1",
            use_cuda=False,
            n_threads=1,
            work_dir=work_dir,
            cold_cuda=False,
        )
        cpu_t4_rec, _ = run_single_pass(
            tag="cpu_t4",
            use_cuda=False,
            n_threads=4,
            work_dir=work_dir,
            cold_cuda=False,
        )

        np.testing.assert_array_equal(
            cpu_t2_cone.lightcone_distances, cuda_warm_t2_cone.lightcone_distances
        )
        lightcone_fields = {
            name: compute_field_diff_stats(
                cpu_t2_cone.lightcones[name], cuda_warm_t2_cone.lightcones[name]
            )
            for name in sorted(cpu_t2_cone.lightcones)
        }
        global_quantities: dict[str, Any] = {}
        if cpu_t2_cone.global_quantities and cuda_warm_t2_cone.global_quantities:
            for name in sorted(cpu_t2_cone.global_quantities):
                stats = compute_field_diff_stats(
                    cpu_t2_cone.global_quantities[name],
                    cuda_warm_t2_cone.global_quantities[name],
                )
                global_quantities[name] = {
                    "max_abs": stats["max_abs"],
                    "rel_rms": stats["rel_rms"],
                }
        ps_comparison = _compute_ps_comparison(cpu_t2_cone, cuda_warm_t2_cone)

    assert "cupy" not in sys.modules
    cuda_warm_t2_compute = cuda_warm_t2_rec["compute_seconds"]
    cuda_warm_t2_total = cuda_warm_t2_rec["total_with_save_seconds"]
    cuda_warm_t1_compute = cuda_warm_t1_rec["compute_seconds"]
    cuda_warm_t1_total = cuda_warm_t1_rec["total_with_save_seconds"]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "spec": {
            "dim": 192,
            "hii_dim": 64,
            "box_len_Mpc": 200.0,
            "nodes": 43,
            "seed": 424242,
            "recomb_model": "inhomogeneous",
            "use_ts_fluct": True,
        },
        "phase0_reference": phase0_ref,
        "runs": {
            "cuda_cold_t2": cuda_cold_t2_rec,
            "cuda_warm_t2": cuda_warm_t2_rec,
            "cuda_warm_t1": cuda_warm_t1_rec,
            "cpu_t2": cpu_t2_rec,
            "cpu_t1": cpu_t1_rec,
            "cpu_t4": cpu_t4_rec,
        },
        "speedups": {
            "cpu_t2_compute_over_cuda_warm_t2_compute": (
                cpu_t2_rec["compute_seconds"] / cuda_warm_t2_compute
            ),
            "cpu_t2_total_over_cuda_warm_t2_total": (
                cpu_t2_rec["total_with_save_seconds"] / cuda_warm_t2_total
            ),
            "cpu_t1_compute_over_cuda_warm_t1_compute": (
                cpu_t1_rec["compute_seconds"] / cuda_warm_t1_compute
            ),
            "cpu_t1_total_over_cuda_warm_t1_total": (
                cpu_t1_rec["total_with_save_seconds"] / cuda_warm_t1_total
            ),
            "cpu_t4_compute_over_cuda_warm_t2_compute": (
                cpu_t4_rec["compute_seconds"] / cuda_warm_t2_compute
            ),
            "cpu_t4_total_over_cuda_warm_t2_total": (
                cpu_t4_rec["total_with_save_seconds"] / cuda_warm_t2_total
            ),
            "phase0_cpu_over_cuda_warm_t2_total": (
                phase0_ref["cpu_phase0_s"] / cuda_warm_t2_total
            ),
            "phase0_cupy_warm_over_cuda_warm_t2_total": (
                phase0_ref["cupy_cufft_warm_s"] / cuda_warm_t2_total
            ),
            "cuda_warm_t2_total_over_phase0_cupy_warm": (
                cuda_warm_t2_total / phase0_ref["cupy_cufft_warm_s"]
            ),
        },
        "accuracy_cuda_warm_t2_vs_cpu_t2": {
            "lightcone_fields": lightcone_fields,
            "global_quantities": global_quantities,
            "brightness_temp_power_3d": ps_comparison,
        },
        "cupy_imported": "cupy" in sys.modules,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["speedups"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
