"""Phase 0 baseline: force 100% single-precision cuFFT across v4_ref without modifying 21cmFAST-GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from unittest.mock import patch


def _configure_paths_and_patch_cufft(v4_dir: Path, upstream_root: Path):
    """Add v4_ref and upstream src/tests to sys.path and replace fft_plan_cuda with cuFFT."""
    for p in (str(v4_dir), str(upstream_root / "src"), str(upstream_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import cupy as cp
    import fft_general
    import fft_plan_cuda

    counts = {
        "cufft_forward_calls": 0,
        "cufft_inverse_calls": 0,
        "plan_cuda_calls": 0,
    }

    def forbid_plan_cuda(*args, **kwargs):
        counts["plan_cuda_calls"] += 1
        raise AssertionError(
            "fft_plan_cuda compiler/executor must not be called in Phase 0 cuFFT mode"
        )

    def cufft_3d(data, shape, inverse=False, threads=2, wisdom=False, strict=False):
        nx, ny, nz = (int(d) for d in shape)
        if inverse:
            counts["cufft_inverse_calls"] += 1
            arr = cp.asarray(data, dtype=cp.complex64, order="C")
            return cp.ascontiguousarray(
                cp.fft.irfftn(arr, s=(nx, ny, nz), norm="forward").astype(
                    cp.float32, copy=False
                )
            )
        counts["cufft_forward_calls"] += 1
        arr = cp.asarray(data, dtype=cp.float32, order="C")
        return cp.ascontiguousarray(cp.fft.rfftn(arr).astype(cp.complex64, copy=False))

    def cufft_cube(a, n, inverse=False, threads=2, *, plan_tree=None):
        dim = int(n)
        return cufft_3d(a, (dim, dim, dim), inverse=inverse, threads=threads)

    fft_plan_cuda.parse_tree = forbid_plan_cuda
    fft_plan_cuda.plans = forbid_plan_cuda
    fft_plan_cuda.compile_plan = forbid_plan_cuda
    fft_plan_cuda.executable = lambda n, inverse, threads: (forbid_plan_cuda, None)
    fft_plan_cuda.transform = cufft_cube
    fft_general.transform_3d = cufft_3d
    return counts


def _diff_stats(cpu_arr, gpu_arr):
    import numpy as np

    x = np.asarray(cpu_arr, dtype=np.float64)
    y = np.asarray(gpu_arr, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: CPU {x.shape} != GPU {y.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Nonfinite values encountered during comparison")
    d = np.abs(y - x)
    nz = np.abs(x) > 1e-6
    rel = d[nz] / np.abs(x[nz])
    bad_strict = d > (1e-8 + 1e-4 * np.abs(x))
    bad_relaxed = d > (1e-5 + 1e-4 * np.abs(x))
    return {
        "count": int(d.size),
        "nonzero_gt_1e6_count": int(nz.sum()),
        "max_abs": float(d.max()) if d.size else 0.0,
        "p99_9_abs": float(np.percentile(d, 99.9)) if d.size else 0.0,
        "max_rel_nonzero": float(rel.max()) if rel.size else 0.0,
        "p99_9_rel_nonzero": float(np.percentile(rel, 99.9)) if rel.size else 0.0,
        "strict_1e8_1e4_failed": int(bad_strict.sum()),
        "strict_1e8_1e4_pct": float(100.0 * bad_strict.sum() / d.size)
        if d.size
        else 0.0,
        "relaxed_1e5_1e4_failed": int(bad_relaxed.sum()),
        "relaxed_1e5_1e4_pct": float(100.0 * bad_relaxed.sum() / d.size)
        if d.size
        else 0.0,
    }


def _ps_stats(cpu_arr, gpu_arr):
    import numpy as np

    base = _diff_stats(cpu_arr, gpu_arr)
    x = np.asarray(cpu_arr, dtype=np.float64)
    y = np.asarray(gpu_arr, dtype=np.float64)
    d = np.abs(y - x)
    nz = x != 0
    pure_rel = np.zeros_like(d)
    pure_rel[nz] = d[nz] / np.abs(x[nz])
    base["rtol_1e4_failed"] = int((d > (1e-8 + 1e-4 * np.abs(x))).sum())
    base["rtol_1e3_failed"] = int((d > (1e-8 + 1e-3 * np.abs(x))).sum())
    base["pure_rtol_1e4_failed"] = int((nz & (pure_rel > 1e-4)).sum())
    base["pure_rtol_1e3_failed"] = int((nz & (pure_rel > 1e-3)).sum())
    return base


def _compute_ps32_on_gpu(cone_gen, target_k, inputs):
    import cupy as cp
    import numpy as np

    sim = inputs.simulation_options
    n, box = sim.HII_DIM, sim.BOX_LEN
    k_vec = cp.fft.fftfreq(n, d=box / n) * 2 * cp.pi
    radii = cp.sqrt(
        k_vec[:, None, None] ** 2
        + k_vec[None, :, None] ** 2
        + k_vec[None, None, :] ** 2
    )
    edges = cp.linspace(0, radii.max() * 1.0000001, 61)
    centers = (edges[:-1] + edges[1:]) / 2
    assert centers[0] < target_k[0] < target_k[-1] < centers[-1]
    bins = cp.clip(cp.digitize(radii.ravel(), edges) - 1, 0, 59)
    count = cp.maximum(cp.bincount(bins, minlength=60), 1)
    factor = cp.asarray(centers**3 / (2 * np.pi**2))
    target_dev = cp.asarray(target_k)

    rows = []
    cone = None
    for _, z, coeval, last_cone in cone_gen:
        cone = last_cone
        if coeval is None:
            continue
        field = cp.asarray(coeval.brightness_temp, dtype=cp.float64)
        assert field.shape == (n, n, n)
        fk = cp.fft.fftn(field - field.mean()) / n**3
        power = cp.abs(fk) ** 2 * box**3
        spectrum = (
            cp.bincount(bins, weights=power.ravel(), minlength=60) / count * factor
        )
        ps = cp.interp(target_dev, centers, spectrum)
        rows.append({"z": float(z), "ps": ps.tolist()})
    return cone, rows


def run_benchmark_1_192(work_dir: Path, ref_dir: Path, point_cfg: Path, counts: dict):
    """Benchmark 1: 192^3 / 64^3 Eulerian lightcone vs archived CPU timing-192."""
    import cupy as cp
    import gpu_lightcone as gl
    import h5py
    import lightcone_pair
    import lightcone_ps
    import numpy as np

    import py21cmfast as p21
    from py21cmfast.drivers._global_initialization import (
        _GlobalInitManagerSingleton as manager,
    )

    cpu_json = json.loads((ref_dir / "cpu.json").read_text())
    cpu_ps_json = json.loads((ref_dir / "cpu-ps.json").read_text())
    target_k = lightcone_ps.k_grid(point_cfg)
    assert np.array_equal(cpu_ps_json["k"], target_k)
    inputs, lightconer = lightcone_pair.benchmark_case(192)
    nodes = list(inputs.node_redshifts)
    assert cpu_json["nodes"] == nodes

    def one_pass(tag: str):
        manager.free()
        diag_dir = work_dir / f"b1_{tag}_diag"
        h5_path = work_dir / f"b1_{tag}.h5"
        c0 = counts["cufft_forward_calls"] + counts["cufft_inverse_calls"]
        t0 = perf_counter()
        with gl._backend(
            inputs, lightconer, diag_dir, fallback_to_cpu=False
        ) as timings:
            ic = p21.compute_initial_conditions(
                inputs=inputs, write=False, regenerate=True
            )
            t_ic = perf_counter()
            cone, rows = _compute_ps32_on_gpu(
                p21.generate_lightcone(
                    lightconer=lightconer,
                    inputs=inputs,
                    initial_conditions=ic,
                    write=False,
                    regenerate=True,
                ),
                target_k,
                inputs,
            )
            cp.cuda.runtime.deviceSynchronize()
            t_compute_end = perf_counter()
            cone.save(h5_path, clobber=True)
            with h5_path.open("rb") as f:
                os.fsync(f.fileno())
            t_end = perf_counter()
        c1 = counts["cufft_forward_calls"] + counts["cufft_inverse_calls"]
        return {
            "ic_seconds": t_ic - t0,
            "lightcone_seconds": t_compute_end - t_ic,
            "compute_seconds": t_compute_end - t0,
            "total_with_save_seconds": t_end - t0,
            "perturb_seconds": float(sum(timings.perturb)),
            "brightness_seconds": float(sum(timings.brightness)),
            "cufft_calls_in_pass": int(c1 - c0),
            "h5_path": h5_path,
            "rows": rows,
            "diag_dir": diag_dir,
        }

    cold = one_pass("cold")
    warm = one_pass("warm")
    assert counts["plan_cuda_calls"] == 0
    assert warm["cufft_calls_in_pass"] > 0

    cpu_elapsed = float(cpu_json["elapsed_seconds"])
    prior_plan_cuda_elapsed = 16.1513898660196

    with h5py.File(ref_dir / "cpu.h5", "r") as ref, h5py.File(
        warm["h5_path"], "r"
    ) as got:
        assert np.array_equal(
            ref["lightcone_distances"][:], got["lightcone_distances"][:]
        )
        lc_fields = {}
        aux_fields = {}
        primary_set = set(lightcone_pair.FIELDS)
        for name in sorted(ref["lightcones"]):
            assert name in got["lightcones"], f"Missing GPU lightcone field {name}"
            stats = _diff_stats(ref["lightcones"][name][:], got["lightcones"][name][:])
            if name in primary_set:
                lc_fields[name] = stats
            else:
                aux_fields[name] = stats
        global_quantities = {}
        for name in sorted(ref["global_quantities"]):
            assert name in got["global_quantities"], f"Missing GPU global {name}"
            global_quantities[name] = _diff_stats(
                ref["global_quantities"][name][:], got["global_quantities"][name][:]
            )

    cpu_ps = np.asarray([r["ps"] for r in cpu_ps_json["rows"]], dtype=np.float64)
    gpu_ps = np.asarray([r["ps"] for r in warm["rows"]], dtype=np.float64)
    ps32_stats = _ps_stats(cpu_ps, gpu_ps)

    return {
        "dim": 192,
        "hii_dim": 64,
        "box_len_Mpc": 200.0,
        "nodes": len(nodes),
        "cpu_elapsed_seconds": cpu_elapsed,
        "prior_fft_plan_cuda_gpu_seconds": prior_plan_cuda_elapsed,
        "gpu_cold_seconds": cold["total_with_save_seconds"],
        "gpu_cold_compute_seconds": cold["compute_seconds"],
        "gpu_warm_seconds": warm["total_with_save_seconds"],
        "gpu_warm_compute_seconds": warm["compute_seconds"],
        "speedup_warm_vs_cpu": cpu_elapsed / warm["total_with_save_seconds"],
        "speedup_cold_vs_cpu": cpu_elapsed / cold["total_with_save_seconds"],
        "ratio_warm_vs_fft_plan_cuda": warm["total_with_save_seconds"]
        / prior_plan_cuda_elapsed,
        "cufft_calls_in_pass": warm["cufft_calls_in_pass"],
        "lightcone_primary_fields": lc_fields,
        "lightcone_auxiliary_fields": aux_fields,
        "global_quantities": global_quantities,
        "ps32": ps32_stats,
    }


def run_benchmark_2_chmf96(work_dir: Path, counts: dict):
    """Benchmark 2: 96^3 / 32^3 thermal CHMF-SAMPLER 1T hybrid CPU vs cuFFT-GPU."""
    import cupy as cp
    import gpu_lightcone as gl
    import numpy as np
    from verify_chmf_lightcone import FIELDS, THERMAL_FIELDS, record_catalogs

    import py21cmfast as p21
    from py21cmfast.drivers._global_initialization import (
        _GlobalInitManagerSingleton as manager,
    )
    from py21cmfast.lightconers import RectilinearLightconer

    nodes = [11.5, 11.0, 10.5, 10.0, 9.5]
    inputs = p21.InputParameters(
        random_seed=81033, node_redshifts=nodes
    ).evolve_input_structs(
        SOURCE_MODEL="CHMF-SAMPLER",
        USE_UPPER_STELLAR_TURNOVER=True,
        DIM=96,
        HII_DIM=32,
        BOX_LEN=84.0,
        N_THREADS=1,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        HII_FILTER="spherical-tophat",
        USE_EXP_FILTER=True,
        CELL_RECOMB=True,
        USE_METALLICITY=True,
        Z_HEAT_MAX=float(nodes[0]),
    )
    lc = RectilinearLightconer.between_redshifts(
        min_redshift=9.6,
        max_redshift=11.3,
        quantities=FIELDS + THERMAL_FIELDS,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )
    env_patch = {"P21_CUDA_GRID_SAMPLE": "1", "P21_CUDA_PROGENITOR": "1"}

    results = {}
    cones = {}
    with patch.dict(os.environ, env_patch):
        for tag in ("cpu", "gpu_cold", "gpu_warm"):
            manager.free()
            cats = []
            c0 = counts["cufft_forward_calls"] + counts["cufft_inverse_calls"]
            with ExitStack() as stack:
                record_catalogs(stack, cats)
                t0 = perf_counter()
                if tag == "cpu":
                    cone = p21.run_lightcone(
                        inputs=inputs, lightconer=lc, write=False, regenerate=True
                    )
                else:
                    cone = gl.run_lightcone(
                        inputs=inputs,
                        lightconer=lc,
                        diagnostics_dir=work_dir / f"b2_{tag}_diag",
                        fallback_to_cpu=False,
                        write=False,
                        regenerate=True,
                    )
                    cp.cuda.runtime.deviceSynchronize()
                elapsed = perf_counter() - t0
            c1 = counts["cufft_forward_calls"] + counts["cufft_inverse_calls"]
            cones[tag] = cone
            results[tag] = {
                "elapsed_seconds": elapsed,
                "catalogs": cats,
                "cufft_calls": int(c1 - c0),
            }

    assert counts["plan_cuda_calls"] == 0
    assert results["gpu_warm"]["cufft_calls"] > 0

    cpu_cone, gpu_cone = cones["cpu"], cones["gpu_warm"]
    assert np.array_equal(
        cpu_cone.lightcone_distances.to_value("Mpc"),
        gpu_cone.lightcone_distances.to_value("Mpc"),
    )

    per_node_halos = []
    for c_row, g_row in zip(
        results["cpu"]["catalogs"], results["gpu_warm"]["catalogs"], strict=True
    ):
        assert abs(c_row["z"] - g_row["z"]) < 1e-6
        exact_cat = c_row["n"] == g_row["n"] and all(
            c_row[k] == g_row[k]
            for k in ("halo_coords", "halo_masses", "star_rng", "sfr_rng", "xray_rng")
        )
        per_node_halos.append(
            {
                "z": float(c_row["z"]),
                "cpu_n_halos": int(c_row["n"]),
                "gpu_n_halos": int(g_row["n"]),
                "delta": int(g_row["n"] - c_row["n"]),
                "exact_equal": bool(exact_cat),
            }
        )

    lc_fields = {
        name: _diff_stats(cpu_cone.lightcones[name], gpu_cone.lightcones[name])
        for name in sorted(cpu_cone.lightcones)
    }
    global_quantities = {
        name: _diff_stats(
            cpu_cone.global_quantities[name], gpu_cone.global_quantities[name]
        )
        for name in sorted(cpu_cone.global_quantities)
    }

    # Transverse brightness PS matching compare_lightcone_ps.compare_saved_slices
    tb_cpu = cpu_cone.lightcones["brightness_temp"].astype(np.float64)
    tb_gpu = gpu_cone.lightcones["brightness_temp"].astype(np.float64)
    side = tb_cpu.shape[0]
    box = float(inputs.simulation_options.BOX_LEN)
    modes = np.fft.fftfreq(side) * side
    radius = np.hypot(modes[:, None], modes[None, :])
    shells = [
        (radius >= i - 0.5) & (radius < i + 0.5) & (radius <= side / 2)
        for i in range(1, side // 2 + 1)
    ]

    def tps(arr):
        centered = arr - arr.mean(axis=(0, 1), keepdims=True)
        pwr = (box / (side * side)) ** 2 * np.abs(
            np.fft.fft2(centered, axes=(0, 1))
        ) ** 2
        return np.stack([pwr[mask].mean(axis=0) for mask in shells])

    transverse_ps = _ps_stats(tps(tb_cpu), tps(tb_gpu))

    return {
        "dim": 96,
        "hii_dim": 32,
        "box_len_Mpc": 84.0,
        "threads": 1,
        "cuda_prefix": False,
        "cpu_seconds": results["cpu"]["elapsed_seconds"],
        "gpu_cold_seconds": results["gpu_cold"]["elapsed_seconds"],
        "gpu_warm_seconds": results["gpu_warm"]["elapsed_seconds"],
        "speedup_warm_vs_cpu": results["cpu"]["elapsed_seconds"]
        / results["gpu_warm"]["elapsed_seconds"],
        "cufft_calls_in_pass": results["gpu_warm"]["cufft_calls"],
        "per_node_halos": per_node_halos,
        "all_catalogs_exact_equal": all(r["exact_equal"] for r in per_node_halos),
        "lightcone_fields": lc_fields,
        "global_quantities": global_quantities,
        "transverse_brightness_ps": transverse_ps,
    }


def run_benchmark_3_upstream670(work_dir: Path, counts: dict):
    """Benchmark 3: Upstream integration configs (simple, no-mdz, ts, minimize_mem, inhomo_ts)."""
    import cupy as cp
    import gpu_lightcone as gl
    import numpy as np
    from powerbox import get_power

    import py21cmfast as p21
    from py21cmfast.drivers._global_initialization import (
        _GlobalInitManagerSingleton as manager,
    )
    from py21cmfast.lightconers import RectilinearLightconer
    from tests import produce_integration_test_data as prd

    selected = ("simple", "no-mdz", "ts", "minimize_mem", "inhomo_ts")
    out = {}
    for name in selected:
        redshift, kwargs = prd.OPTIONS_TESTRUNS[name]
        options = prd.get_all_options_struct(redshift, lc=True, **kwargs)
        inputs = options["inputs"]
        node_z = inputs.node_redshifts
        quantities = prd.get_lc_fields(inputs)
        lcn = RectilinearLightconer.between_redshifts(
            min_redshift=node_z[-1] + 0.2,
            max_redshift=node_z[0] - 0.2,
            quantities=quantities,
            resolution=inputs.simulation_options.cell_size,
        )

        cpu_cache = p21.OutputCache(work_dir / f"b3_{name}_cpu_cache")
        gpu_cache = p21.OutputCache(work_dir / f"b3_{name}_gpu_cache")

        manager.free()
        t0 = perf_counter()
        cpu_cone = p21.run_lightcone(
            lightconer=lcn, inputs=inputs, write=True, regenerate=True, cache=cpu_cache
        )
        cpu_s = perf_counter() - t0

        manager.free()
        c0 = counts["cufft_forward_calls"] + counts["cufft_inverse_calls"]
        t0 = perf_counter()
        gpu_cone = gl.run_lightcone(
            lightconer=lcn,
            inputs=inputs,
            write=True,
            regenerate=True,
            cache=gpu_cache,
            diagnostics_dir=work_dir / f"b3_{name}_diag",
            fallback_to_cpu=False,
        )
        cp.cuda.runtime.deviceSynchronize()
        gpu_s = perf_counter() - t0
        c1 = counts["cufft_forward_calls"] + counts["cufft_inverse_calls"]
        assert counts["plan_cuda_calls"] == 0
        assert c1 > c0

        gq = {}
        upstream_global_rtol1e3_pass = True
        for key in sorted(cpu_cone.global_quantities):
            x = np.asarray(cpu_cone.global_quantities[key], dtype=np.float64)
            y = np.asarray(gpu_cone.global_quantities[key], dtype=np.float64)
            stats = _diff_stats(x, y)
            allclose_1e3 = bool(np.allclose(x, y, atol=0.0, rtol=1e-3))
            stats["upstream_allclose_atol0_rtol1e3"] = allclose_1e3
            upstream_global_rtol1e3_pass &= allclose_1e3
            gq[key] = stats

        lc_stats = {}
        ps_1d = {}
        upstream_ps_1e4_pass = True
        upstream_ps_1e3_pass = True
        for field in prd.LIGHTCONE_FIELDS:
            if field not in cpu_cone.lightcones:
                continue
            c_arr = cpu_cone.lightcones[field]
            g_arr = gpu_cone.lightcones[field]
            lc_stats[field] = _diff_stats(c_arr, g_arr)
            res_c = get_power(
                c_arr, boxlength=cpu_cone.lightcone_dimensions, bins_upto_boxlen=True
            )
            res_g = get_power(
                g_arr, boxlength=gpu_cone.lightcone_dimensions, bins_upto_boxlen=True
            )
            assert np.allclose(res_c.bin_avg, res_g.bin_avg)
            p_stats = _ps_stats(res_c.power, res_g.power)
            p_stats["upstream_allclose_atol0_rtol1e4"] = bool(
                np.allclose(res_c.power, res_g.power, atol=0.0, rtol=1e-4)
            )
            p_stats["allclose_atol0_rtol1e3"] = bool(
                np.allclose(res_c.power, res_g.power, atol=0.0, rtol=1e-3)
            )
            upstream_ps_1e4_pass &= p_stats["upstream_allclose_atol0_rtol1e4"]
            upstream_ps_1e3_pass &= p_stats["allclose_atol0_rtol1e3"]
            ps_1d[field] = p_stats

        out[name] = {
            "source_model": inputs.matter_options.SOURCE_MODEL,
            "use_ts_fluct": bool(inputs.astro_options.USE_TS_FLUCT),
            "recomb_model": inputs.astro_options.RECOMB_MODEL,
            "minimize_memory": bool(inputs.matter_options.MINIMIZE_MEMORY),
            "dim": int(inputs.simulation_options.DIM),
            "hii_dim": int(inputs.simulation_options.HII_DIM),
            "n_nodes": len(node_z),
            "cpu_seconds": cpu_s,
            "gpu_seconds": gpu_s,
            "speedup": cpu_s / gpu_s,
            "cufft_calls": int(c1 - c0),
            "upstream_global_rtol1e3_pass": upstream_global_rtol1e3_pass,
            "upstream_ps1d_rtol1e4_pass": upstream_ps_1e4_pass,
            "ps1d_rtol1e3_pass": upstream_ps_1e3_pass,
            "global_quantities": gq,
            "lightcone_fields": lc_stats,
            "power_spectra_1d": ps_1d,
        }
    return out


def main():
    """Run Phase 0 cuFFT baseline benchmarks and write JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--timing192-ref", type=Path, required=True)
    parser.add_argument("--point-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    counts = _configure_paths_and_patch_cufft(args.v4_dir, args.upstream_root)
    import cupy as cp

    import py21cmfast as p21

    with tempfile.TemporaryDirectory(prefix="phase0-cufft-") as tmp:
        work = Path(tmp)
        b1 = run_benchmark_1_192(work, args.timing192_ref, args.point_config, counts)
        b2 = run_benchmark_2_chmf96(work, counts)
        b3 = run_benchmark_3_upstream670(work, counts)

    assert counts["plan_cuda_calls"] == 0
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "py21cmfast_file": p21.__file__,
        "py21cmfast_version": p21.__version__,
        "cupy_version": cp.__version__,
        "fft_counts": counts,
        "benchmark_1_eulerian_192": b1,
        "benchmark_2_chmf_96": b2,
        "benchmark_3_upstream_integration": b3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output}")
    print(
        f"B1 192^3: cpu={b1['cpu_elapsed_seconds']:.2f}s "
        f"cold={b1['gpu_cold_seconds']:.2f}s warm={b1['gpu_warm_seconds']:.2f}s "
        f"speedup={b1['speedup_warm_vs_cpu']:.2f}x "
        f"Tb_max_abs={b1['lightcone_primary_fields']['brightness_temp']['max_abs']:.3e} "
        f"ps32_max_rel={b1['ps32']['max_rel_nonzero']:.3e} "
        f"ps32_fail(1e-4/1e-3)={b1['ps32']['rtol_1e4_failed']}/{b1['ps32']['rtol_1e3_failed']}",
        flush=True,
    )
    print(
        f"B2 CHMF 96^3: cpu={b2['cpu_seconds']:.2f}s warm={b2['gpu_warm_seconds']:.2f}s "
        f"all_cat_eq={b2['all_catalogs_exact_equal']} "
        f"tps_max_rel={b2['transverse_brightness_ps']['max_rel_nonzero']:.3e}",
        flush=True,
    )
    for k, v in b3.items():
        print(
            f"B3 {k}: cpu={v['cpu_seconds']:.2f}s gpu={v['gpu_seconds']:.2f}s "
            f"global_1e3={v['upstream_global_rtol1e3_pass']} "
            f"ps1d_1e4={v['upstream_ps1d_rtol1e4_pass']} ps1d_1e3={v['ps1d_rtol1e3_pass']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
