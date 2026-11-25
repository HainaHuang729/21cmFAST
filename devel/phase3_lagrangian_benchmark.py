"""Phase 3 benchmark: B2 CHMF-SAMPLER (96^3 / 32^3), L-INTEGRAL (192^3 / 64^3), and DEXM-ESF."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton as manager,
)
from py21cmfast.lightconers import RectilinearLightconer

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    "halo_dexm_runs",
    "halo_sampler_runs",
    "halo_perturb_runs",
    "dndm_table_runs",
)

FIELDS: tuple[str, ...] = (
    "brightness_temp",
    "neutral_fraction",
    "xray_ionised_fraction",
    "kinetic_temp_neutral",
    "spin_temperature",
    "cumulative_recombinations",
)


def snapshot_cuda_stats() -> dict[str, int]:
    """Read the full Phase 3 ``P21CudaStats`` struct into a plain dict."""
    stats = ffi.new("P21CudaStats *")
    lib.p21_cuda_get_stats(stats)
    return {name: int(getattr(stats, name)) for name in STAT_FIELDS}


def _load_phase0_b2(report_path: Path) -> dict[str, float]:
    """Load Phase 0 B2 CHMF-SAMPLER reference timings if present."""
    defaults = {
        "cpu_phase0_s": 24.055403287522495,
        "cupy_cufft_warm_s": 48.62381969764829,
    }
    if not report_path.exists():
        return defaults
    data = json.loads(report_path.read_text())
    b2 = data.get("benchmark_2_chmf_96_thermal", {})
    return {
        "cpu_phase0_s": float(
            b2.get("cpu", {}).get("total_with_save_seconds", defaults["cpu_phase0_s"])
        ),
        "cupy_cufft_warm_s": float(
            b2.get("gpu_warm", {}).get(
                "total_with_save_seconds", defaults["cupy_cufft_warm_s"]
            )
        ),
    }


def _run_lightcone_pass(
    *,
    tag: str,
    inputs: p21.InputParameters,
    lightconer: RectilinearLightconer,
    use_cuda: bool,
    fast_sampler: bool,
    work_dir: Path,
    cold_cuda: bool = False,
) -> tuple[dict[str, Any], p21.LightCone]:
    """Execute one end-to-end ``run_lightcone`` pass with clean timing & CUDA stats."""
    out_dir = work_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "lightcone.h5"

    manager.free()
    if cold_cuda:
        lib.p21_cuda_cleanup()
    else:
        lib.p21_cuda_registry_clear()
    lib.p21_cuda_reset_stats()

    with p21.config.use(
        USE_CUDA=use_cuda,
        CUDA_FAST_HALO_SAMPLER=fast_sampler,
        direc=str(out_dir / "cache"),
    ):
        t0 = perf_counter()
        lc = p21.run_lightcone(
            inputs=inputs,
            lightconer=lightconer,
            regenerate=True,
            write=False,
        )
        t_compute = perf_counter() - t0
        lc.save(h5_path)
        t_total = perf_counter() - t0

    return (
        {
            "tag": tag,
            "use_cuda": use_cuda,
            "fast_halo_sampler": fast_sampler,
            "n_threads": int(inputs.simulation_options.N_THREADS),
            "compute_seconds": t_compute,
            "total_with_save_seconds": t_total,
            "cuda_stats": snapshot_cuda_stats(),
        },
        lc,
    )


def _compare_lightcones(cpu_lc: p21.LightCone, gpu_lc: p21.LightCone) -> dict[str, Any]:
    """Compare lightcone grid fields and global quantities between two runs."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from devel.phase2_eulerian_benchmark import compute_field_diff_stats

    fields = {
        k: compute_field_diff_stats(cpu_lc.lightcones[k], gpu_lc.lightcones[k])
        for k in sorted(cpu_lc.lightcones)
    }
    globals_cmp = {
        k: {
            "max_abs": s["max_abs"],
            "rel_rms": s["rel_rms"],
        }
        for k in sorted(cpu_lc.global_quantities)
        for s in [
            compute_field_diff_stats(
                cpu_lc.global_quantities[k], gpu_lc.global_quantities[k]
            )
        ]
    }
    return {"lightcone_fields": fields, "global_quantities": globals_cmp}


def run_benchmark_b2_chmf96(
    work_dir: Path, phase0_b2: dict[str, float]
) -> dict[str, Any]:
    """Benchmark B2: exact Phase 0 96^3 / 32^3 thermal CHMF-SAMPLER lightcone."""
    nodes = [11.5, 11.0, 10.5, 10.0, 9.5]

    def _make(n_threads: int) -> tuple[p21.InputParameters, RectilinearLightconer]:
        inputs = p21.InputParameters(
            random_seed=81033, node_redshifts=nodes
        ).evolve_input_structs(
            SOURCE_MODEL="CHMF-SAMPLER",
            USE_UPPER_STELLAR_TURNOVER=True,
            DIM=96,
            HII_DIM=32,
            BOX_LEN=84.0,
            N_THREADS=n_threads,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
            R_BUBBLE_MAX=50.0,
            HII_FILTER="spherical-tophat",
            USE_EXP_FILTER=True,
            CELL_RECOMB=True,
            USE_METALLICITY=True,
            Z_HEAT_MAX=float(nodes[0]),
        )
        lc = RectilinearLightconer.between_redshifts(
            min_redshift=9.6,
            max_redshift=11.3,
            quantities=FIELDS,
            resolution=inputs.simulation_options.cell_size,
            cosmo=inputs.cosmo_params.cosmo,
        )
        return inputs, lc

    in_t1, lc_t1 = _make(1)
    in_t2, lc_t2 = _make(2)

    # Warm-up CUDA runtime/cuFFT cache briefly
    _run_lightcone_pass(
        tag="b2_warmup",
        inputs=in_t2,
        lightconer=lc_t2,
        use_cuda=True,
        fast_sampler=False,
        work_dir=work_dir,
    )

    cpu_t1_rec, cpu_t1_lc = _run_lightcone_pass(
        tag="b2_cpu_t1",
        inputs=in_t1,
        lightconer=lc_t1,
        use_cuda=False,
        fast_sampler=False,
        work_dir=work_dir,
    )
    cpu_t2_rec, _ = _run_lightcone_pass(
        tag="b2_cpu_t2",
        inputs=in_t2,
        lightconer=lc_t2,
        use_cuda=False,
        fast_sampler=False,
        work_dir=work_dir,
    )
    cuda_default_t1_rec, cuda_default_t1_lc = _run_lightcone_pass(
        tag="b2_cuda_default_t1",
        inputs=in_t1,
        lightconer=lc_t1,
        use_cuda=True,
        fast_sampler=False,
        work_dir=work_dir,
    )
    cuda_default_t2_rec, _ = _run_lightcone_pass(
        tag="b2_cuda_default_t2",
        inputs=in_t2,
        lightconer=lc_t2,
        use_cuda=True,
        fast_sampler=False,
        work_dir=work_dir,
    )
    cuda_fast_t2_rec, cuda_fast_t2_lc = _run_lightcone_pass(
        tag="b2_cuda_fast_t2",
        inputs=in_t2,
        lightconer=lc_t2,
        use_cuda=True,
        fast_sampler=True,
        work_dir=work_dir,
    )

    return {
        "spec": {
            "dim": 96,
            "hii_dim": 32,
            "box_len_Mpc": 84.0,
            "nodes": nodes,
            "random_seed": 81033,
            "source_model": "CHMF-SAMPLER",
        },
        "phase0_reference": phase0_b2,
        "runs": {
            "cpu_t1": cpu_t1_rec,
            "cpu_t2": cpu_t2_rec,
            "cuda_default_t1": cuda_default_t1_rec,
            "cuda_default_t2": cuda_default_t2_rec,
            "cuda_fast_t2": cuda_fast_t2_rec,
        },
        "speedups": {
            "cpu_t1_over_cuda_default_t1": (
                cpu_t1_rec["total_with_save_seconds"]
                / cuda_default_t1_rec["total_with_save_seconds"]
            ),
            "cpu_t2_over_cuda_default_t2": (
                cpu_t2_rec["total_with_save_seconds"]
                / cuda_default_t2_rec["total_with_save_seconds"]
            ),
            "cpu_t2_over_cuda_fast_t2": (
                cpu_t2_rec["total_with_save_seconds"]
                / cuda_fast_t2_rec["total_with_save_seconds"]
            ),
            "phase0_cupy_warm_over_cuda_default_t1": (
                phase0_b2["cupy_cufft_warm_s"]
                / cuda_default_t1_rec["total_with_save_seconds"]
            ),
            "phase0_cupy_warm_over_cuda_fast_t2": (
                phase0_b2["cupy_cufft_warm_s"]
                / cuda_fast_t2_rec["total_with_save_seconds"]
            ),
        },
        "accuracy_cuda_default_t1_vs_cpu_t1": _compare_lightcones(
            cpu_t1_lc, cuda_default_t1_lc
        ),
        "global_means_cuda_fast_t2_vs_cpu_t1": {
            k: {
                "cpu_t1_mean": float(np.mean(cpu_t1_lc.global_quantities[k])),
                "cuda_fast_t2_mean": float(
                    np.mean(cuda_fast_t2_lc.global_quantities[k])
                ),
            }
            for k in sorted(cpu_t1_lc.global_quantities)
        },
    }


def run_benchmark_l_integral_192(work_dir: Path) -> dict[str, Any]:
    """Benchmark L-INTEGRAL (192^3 / 64^3, 10 nodes): CPU 2T vs CUDA warm 2T."""
    base = p21.InputParameters(random_seed=424242).with_logspaced_redshifts(
        zmin=7.5, zmax=15.0, step=1.08
    )
    # Keep first 10 nodes
    nodes = list(base.node_redshifts[:10])
    inputs = p21.InputParameters(
        random_seed=424242, node_redshifts=nodes
    ).evolve_input_structs(
        DIM=192,
        HII_DIM=64,
        BOX_LEN=200.0,
        N_THREADS=2,
        Z_HEAT_MAX=15.0,
        SOURCE_MODEL="L-INTEGRAL",
        USE_UPPER_STELLAR_TURNOVER=False,
        USE_TS_FLUCT=True,
        USE_METALLICITY=True,
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
    )
    lc = RectilinearLightconer.between_redshifts(
        min_redshift=float(nodes[-1]) + 0.1,
        max_redshift=float(nodes[0]) - 0.1,
        quantities=FIELDS,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )

    cuda_rec, cuda_lc = _run_lightcone_pass(
        tag="lint_cuda_t2",
        inputs=inputs,
        lightconer=lc,
        use_cuda=True,
        fast_sampler=False,
        work_dir=work_dir,
    )
    cpu_rec, cpu_lc = _run_lightcone_pass(
        tag="lint_cpu_t2",
        inputs=inputs,
        lightconer=lc,
        use_cuda=False,
        fast_sampler=False,
        work_dir=work_dir,
    )
    return {
        "spec": {
            "dim": 192,
            "hii_dim": 64,
            "box_len_Mpc": 200.0,
            "nodes": len(nodes),
            "source_model": "L-INTEGRAL",
        },
        "runs": {"cpu_t2": cpu_rec, "cuda_warm_t2": cuda_rec},
        "speedups": {
            "cpu_t2_total_over_cuda_warm_t2_total": (
                cpu_rec["total_with_save_seconds"] / cuda_rec["total_with_save_seconds"]
            ),
            "cpu_t2_compute_over_cuda_warm_t2_compute": (
                cpu_rec["compute_seconds"] / cuda_rec["compute_seconds"]
            ),
        },
        "accuracy_cuda_vs_cpu": _compare_lightcones(cpu_lc, cuda_lc),
    }


def run_benchmark_dexm_96(work_dir: Path) -> dict[str, Any]:
    """Benchmark DEXM-ESF (96^3 / 32^3, 3 nodes): CPU 2T vs CUDA warm 2T."""
    nodes = [10.0, 9.0, 8.0]
    inputs = p21.InputParameters(
        random_seed=424242, node_redshifts=nodes
    ).evolve_input_structs(
        DIM=96,
        HII_DIM=32,
        BOX_LEN=84.0,
        N_THREADS=2,
        Z_HEAT_MAX=float(nodes[0]),
        SOURCE_MODEL="DEXM-ESF",
        DEXM_OPTIMIZE=True,
        USE_TS_FLUCT=True,
        USE_UPPER_STELLAR_TURNOVER=True,
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=40.0,
    )
    lc = RectilinearLightconer.between_redshifts(
        min_redshift=8.1,
        max_redshift=9.9,
        quantities=("brightness_temp", "neutral_fraction", "spin_temperature"),
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )

    cuda_rec, cuda_lc = _run_lightcone_pass(
        tag="dexm_cuda_t2",
        inputs=inputs,
        lightconer=lc,
        use_cuda=True,
        fast_sampler=False,
        work_dir=work_dir,
    )
    cpu_rec, cpu_lc = _run_lightcone_pass(
        tag="dexm_cpu_t2",
        inputs=inputs,
        lightconer=lc,
        use_cuda=False,
        fast_sampler=False,
        work_dir=work_dir,
    )
    return {
        "spec": {
            "dim": 96,
            "hii_dim": 32,
            "box_len_Mpc": 84.0,
            "nodes": nodes,
            "source_model": "DEXM-ESF",
        },
        "runs": {"cpu_t2": cpu_rec, "cuda_warm_t2": cuda_rec},
        "speedups": {
            "cpu_t2_total_over_cuda_warm_t2_total": (
                cpu_rec["total_with_save_seconds"] / cuda_rec["total_with_save_seconds"]
            ),
            "cpu_t2_compute_over_cuda_warm_t2_compute": (
                cpu_rec["compute_seconds"] / cuda_rec["compute_seconds"]
            ),
        },
        "accuracy_cuda_vs_cpu": _compare_lightcones(cpu_lc, cuda_lc),
    }


def main() -> None:
    """Run all Phase 3 Lagrangian & Halo benchmarks and write JSON report."""
    assert "cupy" not in sys.modules
    if lib.p21_cuda_is_compiled() != 1 or lib.p21_cuda_is_available() != 1:
        raise RuntimeError(
            "Native CUDA runtime must be compiled and available to run Phase 3 benchmark."
        )

    out_path = REPO_ROOT / "reports" / "phase3-lagrangian-benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phase0_b2 = _load_phase0_b2(REPO_ROOT / "reports" / "phase0-cufft-baseline.json")

    with tempfile.TemporaryDirectory(prefix="p21_phase3_bench_") as tmp_dir:
        work_dir = Path(tmp_dir)
        b2 = run_benchmark_b2_chmf96(work_dir, phase0_b2)
        lint = run_benchmark_l_integral_192(work_dir)
        dexm = run_benchmark_dexm_96(work_dir)

    assert "cupy" not in sys.modules
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "b2_chmf_96_thermal": b2,
        "l_integral_192": lint,
        "dexm_esf_96": dexm,
        "cupy_imported": "cupy" in sys.modules,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("B2 speedups:", json.dumps(b2["speedups"], indent=2))
    print("L-INTEGRAL speedups:", json.dumps(lint["speedups"], indent=2))
    print("DEXM-ESF speedups:", json.dumps(dexm["speedups"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
