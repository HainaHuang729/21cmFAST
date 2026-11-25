"""Phase 1 tests for the native C/CUDA and cuFFT backend."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi, lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def _make_inputs(
    *,
    random_seed: int,
    dim: int,
    hii_dim: int,
    box_len: float,
    n_threads: int = 2,
    non_cubic_factor: float = 1.0,
    power_spectrum: str = "EH",
    v_cb_model: str = "NONE",
    perturb_algorithm: str = "2LPT",
    perturb_on_high_res: bool = False,
    smooth_evolved: bool = False,
    keep_3d_vel: bool = False,
    use_wisdom: bool = False,
) -> p21.InputParameters:
    """Construct validated inputs for Phase 1 IC and PerturbedField tests."""
    base = p21.InputParameters(random_seed=random_seed)
    extra: dict[str, Any] = {}
    if v_cb_model == "FLUCTS":
        base = base.with_logspaced_redshifts()
        extra.update(
            USE_MCGS=True,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
            R_BUBBLE_MAX=50.0,
            M_TURN_STELLAR_FEEDBACK=5.0,
        )
    return base.evolve_input_structs(
        DIM=dim,
        HII_DIM=hii_dim,
        BOX_LEN=box_len,
        NON_CUBIC_FACTOR=non_cubic_factor,
        N_THREADS=n_threads,
        POWER_SPECTRUM=power_spectrum,
        V_CB_MODEL=v_cb_model,
        PERTURB_ALGORITHM=perturb_algorithm,
        PERTURB_ON_HIGH_RES=perturb_on_high_res,
        SMOOTH_EVOLVED_DENSITY_FIELD=smooth_evolved,
        KEEP_3D_VELOCITIES=keep_3d_vel,
        USE_FFTW_WISDOM=use_wisdom,
        **extra,
    )


def _get_stats() -> Any:
    """Fetch a snapshot of the backend ``P21CudaStats`` struct."""
    stats = ffi.new("P21CudaStats *")
    lib.p21_cuda_get_stats(stats)
    return stats


def _run_ic_cpu_and_gpu(
    tmp_path: Path,
    *,
    inputs: p21.InputParameters,
    initial_density: np.ndarray | None = None,
) -> tuple[p21.InitialConditions, p21.InitialConditions]:
    """Compute initial conditions with both CPU and CUDA backends."""
    with p21.config.use(USE_CUDA=False, direc=str(tmp_path / "cpu")):
        cpu_ic = p21.compute_initial_conditions(
            inputs=inputs,
            initial_density=initial_density,
            regenerate=True,
            write=False,
        )
    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "gpu")):
        before_ic = _get_stats().ic_runs
        gpu_ic = p21.compute_initial_conditions(
            inputs=inputs,
            initial_density=initial_density,
            regenerate=True,
            write=False,
        )
        if inputs.simulation_options.DIM > 1:
            assert _get_stats().ic_runs == before_ic + 1
    return cpu_ic, gpu_ic


def _run_perturb_cpu_and_gpu(
    tmp_path: Path,
    *,
    inputs: p21.InputParameters,
    redshift: float,
    ic_cpu: p21.InitialConditions,
    ic_gpu: p21.InitialConditions,
) -> tuple[p21.PerturbedField, p21.PerturbedField]:
    """Compute perturbed fields with both CPU and CUDA backends."""
    with p21.config.use(USE_CUDA=False, direc=str(tmp_path / "cpu")):
        cpu_pt = p21.perturb_field(
            redshift=redshift,
            inputs=inputs,
            initial_conditions=ic_cpu,
            regenerate=True,
            write=False,
        )
    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "gpu")):
        before_pt = _get_stats().perturb_runs
        gpu_pt = p21.perturb_field(
            redshift=redshift,
            inputs=inputs,
            initial_conditions=ic_gpu,
            regenerate=True,
            write=False,
        )
        if inputs.simulation_options.HII_DIM > 1:
            assert _get_stats().perturb_runs == before_pt + 1
    return cpu_pt, gpu_pt


def _assert_close_field(gpu: np.ndarray, cpu: np.ndarray, name: str) -> None:
    """Assert relative RMS and pointwise closeness between GPU and CPU 3D fields."""
    assert gpu.shape == cpu.shape, f"Shape mismatch for {name}"
    assert np.all(np.isfinite(gpu)), f"Non-finite values in GPU {name}"
    diff = gpu.astype(np.float64) - cpu.astype(np.float64)
    cpu_rms = float(np.sqrt(np.mean(cpu.astype(np.float64) ** 2)))
    diff_rms = float(np.sqrt(np.mean(diff**2)))
    scale = max(cpu_rms, 1.0)
    assert (
        diff_rms / scale < 2e-5
    ), f"{name}: RMS diff {diff_rms:.3e} exceeds tolerance for scale {scale:.3e}"
    np.testing.assert_allclose(gpu, cpu, rtol=5e-4, atol=5e-4 * scale, err_msg=name)


def test_cuda_config_toggle_and_no_cupy() -> None:
    """Verify ``USE_CUDA`` config toggle, ``cupy`` absence, and stats struct."""
    assert "cupy" not in sys.modules
    prev = p21.config["USE_CUDA"]
    try:
        with p21.config.use(USE_CUDA=False):
            assert lib.p21_cuda_get_enabled() == 0
        with p21.config.use(USE_CUDA=True):
            assert lib.p21_cuda_get_enabled() == 1
        assert lib.p21_cuda_get_enabled() == int(bool(prev))
    finally:
        p21.config["USE_CUDA"] = prev
    stats = _get_stats()
    assert stats.compiled == lib.p21_cuda_is_compiled()


@requires_cuda_runtime
@pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
def test_ic_all_gsl_prngs_threads(tmp_path: Path, n_threads: int) -> None:
    """Match CPU GSL PRNGs (mt19937, gfsr4, cmrg, mrg, taus2) across thread counts."""
    assert "cupy" not in sys.modules
    inputs = _make_inputs(
        random_seed=42,
        dim=32,
        hii_dim=16,
        box_len=64.0,
        n_threads=n_threads,
        power_spectrum="EH",
        perturb_algorithm="2LPT",
    )
    lib.p21_cuda_reset_stats()
    cpu_ic, gpu_ic = _run_ic_cpu_and_gpu(tmp_path, inputs=inputs)

    for name in (
        "hires_density",
        "lowres_density",
        "lowres_vx",
        "lowres_vy",
        "lowres_vz",
        "lowres_vx_2LPT",
        "lowres_vy_2LPT",
        "lowres_vz_2LPT",
    ):
        _assert_close_field(
            gpu_ic.get(name),
            cpu_ic.get(name),
            f"IC[threads={n_threads}].{name}",
        )

    stats = _get_stats()
    assert stats.ic_runs >= 1
    assert "cupy" not in sys.modules


@requires_cuda_runtime
@pytest.mark.parametrize(
    ("power_spectrum", "v_cb_model"),
    [
        ("EH", "NONE"),
        ("BBKS", "NONE"),
        ("EFSTATHIOU", "NONE"),
        ("PEEBLES", "NONE"),
        ("WHITE", "NONE"),
        ("CLASS", "NONE"),
        ("CLASS", "FLUCTS"),
    ],
)
def test_ic_power_spectrum_models(
    tmp_path: Path, power_spectrum: str, v_cb_model: str
) -> None:
    """Match CPU across all six ``POWER_SPECTRUM`` models plus ``V_CB_MODEL=FLUCTS``."""
    inputs = _make_inputs(
        random_seed=123,
        dim=32,
        hii_dim=16,
        box_len=80.0,
        n_threads=2,
        power_spectrum=power_spectrum,
        v_cb_model=v_cb_model,
        perturb_algorithm="ZELDOVICH",
    )
    cpu_ic, gpu_ic = _run_ic_cpu_and_gpu(tmp_path, inputs=inputs)
    _assert_close_field(
        gpu_ic.hires_density,
        cpu_ic.hires_density,
        f"IC[{power_spectrum}/{v_cb_model}].hires_density",
    )
    _assert_close_field(
        gpu_ic.lowres_density,
        cpu_ic.lowres_density,
        f"IC[{power_spectrum}/{v_cb_model}].lowres_density",
    )
    if v_cb_model == "FLUCTS":
        _assert_close_field(
            gpu_ic.lowres_vcb,
            cpu_ic.lowres_vcb,
            "IC[CLASS/FLUCTS].lowres_vcb",
        )


@requires_cuda_runtime
def test_ic_external_initial_density(tmp_path: Path) -> None:
    """Match CPU when an external non-zero ``hires_density`` is provided."""
    inputs = _make_inputs(
        random_seed=7,
        dim=24,
        hii_dim=12,
        box_len=48.0,
        n_threads=2,
        perturb_algorithm="2LPT",
    )
    rng = np.random.default_rng(999)
    ext = rng.normal(0.0, 0.2, size=(24, 24, 24)).astype(np.float32)
    ext -= ext.mean()

    cpu_ic, gpu_ic = _run_ic_cpu_and_gpu(tmp_path, inputs=inputs, initial_density=ext)
    for name in ("hires_density", "lowres_density", "lowres_vx", "lowres_vx_2LPT"):
        _assert_close_field(
            gpu_ic.get(name),
            cpu_ic.get(name),
            f"IC_external.{name}",
        )


@requires_cuda_runtime
@pytest.mark.parametrize("perturb_algorithm", ["LINEAR", "ZELDOVICH", "2LPT"])
@pytest.mark.parametrize("perturb_on_high_res", [False, True])
@pytest.mark.parametrize("smooth_evolved", [False, True])
@pytest.mark.parametrize("keep_3d_vel", [False, True])
def test_perturb_algorithm_matrix(
    tmp_path: Path,
    perturb_algorithm: str,
    perturb_on_high_res: bool,
    smooth_evolved: bool,
    keep_3d_vel: bool,
) -> None:
    """Verify IC -> PerturbedField across perturbation algorithms and options."""
    inputs = _make_inputs(
        random_seed=314,
        dim=24,
        hii_dim=12,
        box_len=48.0,
        n_threads=2,
        perturb_algorithm=perturb_algorithm,
        perturb_on_high_res=perturb_on_high_res,
        smooth_evolved=smooth_evolved,
        keep_3d_vel=keep_3d_vel,
    )
    cpu_ic, gpu_ic = _run_ic_cpu_and_gpu(tmp_path, inputs=inputs)
    cpu_pt, gpu_pt = _run_perturb_cpu_and_gpu(
        tmp_path,
        inputs=inputs,
        redshift=8.0,
        ic_cpu=cpu_ic,
        ic_gpu=gpu_ic,
    )

    tag = (
        f"PT[{perturb_algorithm},hi={perturb_on_high_res},"
        f"sm={smooth_evolved},v3d={keep_3d_vel}]"
    )
    _assert_close_field(gpu_pt.density, cpu_pt.density, f"{tag}.density")
    _assert_close_field(gpu_pt.velocity_z, cpu_pt.velocity_z, f"{tag}.velocity_z")
    if keep_3d_vel:
        _assert_close_field(gpu_pt.velocity_x, cpu_pt.velocity_x, f"{tag}.velocity_x")
        _assert_close_field(gpu_pt.velocity_y, cpu_pt.velocity_y, f"{tag}.velocity_y")


@requires_cuda_runtime
@pytest.mark.parametrize(
    ("dim", "hii_dim", "ncf", "use_wisdom"),
    [
        (24, 12, 1.5, False),
        (33, 11, 1.0, False),
        (24, 12, 1.0, True),
        (64, 32, 1.0, False),
    ],
)
def test_grid_edge_cases(
    tmp_path: Path, dim: int, hii_dim: int, ncf: float, use_wisdom: bool
) -> None:
    """Verify non-cubic boxes, odd grid dimensions, and ``USE_FFTW_WISDOM``."""
    inputs = _make_inputs(
        random_seed=2025,
        dim=dim,
        hii_dim=hii_dim,
        box_len=60.0,
        non_cubic_factor=ncf,
        n_threads=2,
        perturb_algorithm="2LPT",
        keep_3d_vel=True,
        use_wisdom=use_wisdom,
    )
    with p21.config.use(wisdoms_path=str(tmp_path / "wisdoms")):
        cpu_ic, gpu_ic = _run_ic_cpu_and_gpu(tmp_path, inputs=inputs)
        cpu_pt, gpu_pt = _run_perturb_cpu_and_gpu(
            tmp_path,
            inputs=inputs,
            redshift=7.5,
            ic_cpu=cpu_ic,
            ic_gpu=gpu_ic,
        )

    _assert_close_field(gpu_ic.hires_density, cpu_ic.hires_density, "grid.ic")
    _assert_close_field(gpu_pt.density, cpu_pt.density, "grid.density")
    _assert_close_field(gpu_pt.velocity_z, cpu_pt.velocity_z, "grid.vz")


@requires_cuda_runtime
def test_cuda_1cell_cpu_fallback(tmp_path: Path) -> None:
    """Verify ``HII_DIM=1`` falls back cleanly to CPU without advancing ``perturb_runs``."""
    inputs = _make_inputs(
        random_seed=11,
        dim=1,
        hii_dim=1,
        box_len=30.0,
        n_threads=1,
    )
    lib.p21_cuda_reset_stats()
    cpu_ic, gpu_ic = _run_ic_cpu_and_gpu(tmp_path, inputs=inputs)
    cpu_pt, gpu_pt = _run_perturb_cpu_and_gpu(
        tmp_path,
        inputs=inputs,
        redshift=10.0,
        ic_cpu=cpu_ic,
        ic_gpu=gpu_ic,
    )
    np.testing.assert_allclose(gpu_pt.density, cpu_pt.density)
    stats = _get_stats()
    assert stats.perturb_runs == 0


@requires_cuda_runtime
def test_stats_registry_hits_and_aba_invalidation(tmp_path: Path) -> None:
    """Verify ``DeviceFieldRegistry`` hits, Array invalidation, and mutation safety."""
    inputs = _make_inputs(
        random_seed=777,
        dim=24,
        hii_dim=12,
        box_len=48.0,
        n_threads=2,
        perturb_algorithm="2LPT",
    )

    lib.p21_cuda_cleanup()
    lib.p21_cuda_reset_stats()

    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "gpu")):
        ic = p21.compute_initial_conditions(inputs=inputs, regenerate=True, write=False)
        st_after_ic = _get_stats()
        assert st_after_ic.ic_runs == 1
        assert st_after_ic.cufft_exec_r2c >= 1
        assert st_after_ic.cufft_exec_c2r >= 8
        assert st_after_ic.mempool_alloc_calls >= 1

        pt1 = p21.perturb_field(
            redshift=9.0,
            inputs=inputs,
            initial_conditions=ic,
            regenerate=True,
            write=False,
        )
        st_after_pt1 = _get_stats()
        assert st_after_pt1.perturb_runs == 1
        assert st_after_pt1.registry_hits >= 7
        assert st_after_pt1.registry_misses == 0

        # Repeat with same IC at another redshift to exercise cuFFT plan hits.
        _pt2 = p21.perturb_field(
            redshift=8.0,
            inputs=inputs,
            initial_conditions=ic,
            regenerate=True,
            write=False,
        )
        st_after_pt2 = _get_stats()
        assert st_after_pt2.perturb_runs == 2
        assert st_after_pt2.cufft_plan_hits >= 1
        assert st_after_pt2.registry_hits > st_after_pt1.registry_hits

        # Mutate IC host array in place (simulates ABA / dirty buffer at same pointer).
        ic.hires_density[3, 4, 5] += 2.5
        inv_before = st_after_pt2.registry_invalidations
        pt_mut_gpu = p21.perturb_field(
            redshift=9.0,
            inputs=inputs,
            initial_conditions=ic,
            regenerate=True,
            write=False,
        )
        st_after_mut = _get_stats()
        assert st_after_mut.registry_invalidations > inv_before
        assert st_after_mut.registry_misses >= 1

    with p21.config.use(USE_CUDA=False, direc=str(tmp_path / "cpu")):
        pt_mut_cpu = p21.perturb_field(
            redshift=9.0,
            inputs=inputs,
            initial_conditions=ic,
            regenerate=True,
            write=False,
        )
    _assert_close_field(pt_mut_gpu.density, pt_mut_cpu.density, "ABA_mutated.density")
    assert not np.allclose(pt_mut_gpu.density, pt1.density)

    # Explicit invalidation via Array.without_value() must trigger registry invalidation.
    inv_before_drop = _get_stats().registry_invalidations
    dropped = ic.arrays["hires_density"].without_value()
    assert dropped._value is None
    assert _get_stats().registry_invalidations > inv_before_drop

    del ic, pt1, pt_mut_gpu, pt_mut_cpu
    gc.collect()
