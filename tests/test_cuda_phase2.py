"""Phase 2 tests for the native C/CUDA Eulerian pipeline."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from powerbox import get_power

import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.drivers.single_field import setup_radiation_fields

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def _get_stats() -> Any:
    """Return a snapshot of the backend ``P21CudaStats`` struct."""
    stats = ffi.new("P21CudaStats *")
    lib.p21_cuda_get_stats(stats)
    return stats


def _make_eulerian_inputs(
    *,
    random_seed: int = 202601,
    dim: int = 48,
    hii_dim: int = 16,
    box_len: float = 64.0,
    non_cubic_factor: float = 1.0,
    n_threads: int = 1,
    min_redshift: float = 8.0,
    max_redshift: float | None = 25.0,
    z_step_factor: float = 1.15,
    source_model: str = "E-INTEGRAL",
    use_ts_fluct: bool = False,
    use_xray_heating: bool = True,
    use_cmb_heating: bool = False,
    use_lya_heating: bool = False,
    lya_multiple_scattering: bool = False,
    use_mcgs: bool = False,
    use_reion_fb: bool = False,
    recomb_model: str = "none",
    cell_recomb: bool = False,
    ionise_entire_sphere: bool = False,
    hii_filter: str = "spherical-tophat",
    photon_cons_type: str = "no-photoncons",
    minimize_memory: bool = False,
    use_metallicity: bool = False,
    halo_median: bool = False,
    use_adiabatic_fluct: bool = True,
    use_exp_filter: bool = False,
    power_spectrum: str = "EH",
    v_cb_model: str = "NONE",
    hii_eff_factor: float | None = None,
) -> p21.InputParameters:
    """Construct validated ``InputParameters`` for Phase 2 Eulerian CUDA tests."""
    base = p21.InputParameters(random_seed=random_seed)
    if max_redshift is not None:
        base = base.with_logspaced_redshifts(
            zmin=min_redshift,
            zmax=max_redshift,
            step=z_step_factor,
        )
    extra: dict[str, Any] = {}
    if use_mcgs:
        extra["M_TURN_STELLAR_FEEDBACK"] = 5.0
    if recomb_model != "none":
        extra["R_BUBBLE_MAX"] = 50.0 if box_len >= 50.0 else box_len
    if max_redshift is not None:
        extra["Z_HEAT_MAX"] = max_redshift
    if hii_eff_factor is not None:
        extra["HII_EFF_FACTOR"] = hii_eff_factor
    if power_spectrum == "CLASS":
        extra["K_MAX_FOR_CLASS"] = 1.0

    return base.evolve_input_structs(
        DIM=dim,
        HII_DIM=hii_dim,
        BOX_LEN=box_len,
        NON_CUBIC_FACTOR=non_cubic_factor,
        N_THREADS=n_threads,
        ZPRIME_STEP_FACTOR=z_step_factor,
        SOURCE_MODEL=source_model,
        POWER_SPECTRUM=power_spectrum,
        V_CB_MODEL=v_cb_model,
        USE_TS_FLUCT=use_ts_fluct,
        USE_X_RAY_HEATING=use_xray_heating,
        USE_CMB_HEATING=use_cmb_heating,
        USE_LYA_HEATING=use_lya_heating,
        LYA_MULTIPLE_SCATTERING=lya_multiple_scattering,
        USE_MCGS=use_mcgs,
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=use_reion_fb,
        RECOMB_MODEL=recomb_model,
        CELL_RECOMB=cell_recomb,
        IONISE_ENTIRE_SPHERE=ionise_entire_sphere,
        HII_FILTER=hii_filter,
        PHOTON_CONS_TYPE=photon_cons_type,
        MINIMIZE_MEMORY=minimize_memory,
        USE_METALLICITY=use_metallicity,
        USE_UPPER_STELLAR_TURNOVER=False,
        HALO_SCALING_RELATIONS_MEDIAN=halo_median,
        USE_ADIABATIC_FLUCTUATIONS=use_adiabatic_fluct,
        USE_EXP_FILTER=use_exp_filter,
        **extra,
    )


def _assert_rel_close(
    gpu: np.ndarray,
    cpu: np.ndarray,
    name: str,
    *,
    rel_rms_tol: float = 1e-4,
    rtol: float = 5e-4,
    atol_scale: float = 1e-4,
) -> None:
    """Assert relative RMS and elementwise closeness between GPU and CPU arrays."""
    assert (
        gpu.shape == cpu.shape
    ), f"Shape mismatch for {name}: {gpu.shape} vs {cpu.shape}"
    assert np.all(np.isfinite(gpu)), f"Non-finite values in GPU {name}"
    g64 = gpu.astype(np.float64)
    c64 = cpu.astype(np.float64)
    cpu_rms = float(np.sqrt(np.mean(c64**2)))
    diff_rms = float(np.sqrt(np.mean((g64 - c64) ** 2)))
    scale = max(cpu_rms, 1e-30)
    if cpu_rms > 0.0:
        assert (
            diff_rms / scale < rel_rms_tol
        ), f"{name}: rel_rms {diff_rms / scale:.3e} >= {rel_rms_tol:.3e}"
    np.testing.assert_allclose(
        g64, c64, rtol=rtol, atol=atol_scale * scale, err_msg=name
    )


def test_phase2_stats_fields_exist_and_no_cupy() -> None:
    """Verify Phase 2 counters exist in ``P21CudaStats`` and ``cupy`` is never imported."""
    assert "cupy" not in sys.modules
    lib.p21_cuda_reset_stats()
    stats = _get_stats()
    for attr in (
        "emissivity_runs",
        "radiation_setup_runs",
        "radiation_update_runs",
        "spin_temp_runs",
        "ionization_runs",
        "brightness_runs",
    ):
        assert hasattr(stats, attr)
        assert getattr(stats, attr) == 0


@requires_cuda_runtime
@pytest.mark.parametrize(
    ("use_ts_fluct", "non_cubic_factor"),
    [(False, 1.0), (True, 1.0), (True, 1.5)],
)
def test_cuda_brightness_temperature(
    tmp_path: Path, use_ts_fluct: bool, non_cubic_factor: float
) -> None:
    """Compare CUDA and CPU ``ComputeBrightnessTemp`` with and without Ts fluctuations."""
    inputs = _make_eulerian_inputs(
        random_seed=11,
        dim=48,
        hii_dim=16,
        box_len=64.0,
        non_cubic_factor=non_cubic_factor,
        use_ts_fluct=use_ts_fluct,
        recomb_model="none",
        max_redshift=20.0 if use_ts_fluct else None,
    )
    rng = np.random.default_rng(123)
    shape = (16, 16, int(round(16 * non_cubic_factor)))
    dens = rng.uniform(-0.8, 2.5, size=shape).astype(np.float32)
    xhi = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
    ts = rng.uniform(2.0, 500.0, size=shape).astype(np.float32)

    pf = p21.PerturbedField.new(inputs=inputs, redshift=9.0)
    pf.set("density", dens)
    ib = p21.IonizedBox.new(inputs=inputs, redshift=9.0)
    ib.set("neutral_fraction", xhi)
    tb_sp = None
    if use_ts_fluct:
        tb_sp = p21.TsBox.new(inputs=inputs, redshift=9.0)
        tb_sp.set("spin_temperature", ts)

    with p21.config.use(USE_CUDA=False, direc=str(tmp_path / "cpu")):
        cpu_bt = p21.brightness_temperature(
            ionized_box=ib,
            perturbed_field=pf,
            spin_temp=tb_sp,
            regenerate=True,
            write=False,
        )
    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "gpu")):
        before = _get_stats().brightness_runs
        gpu_bt = p21.brightness_temperature(
            ionized_box=ib,
            perturbed_field=pf,
            spin_temp=tb_sp,
            regenerate=True,
            write=False,
        )
        assert _get_stats().brightness_runs == before + 1

    _assert_rel_close(
        gpu_bt.get("brightness_temp"),
        cpu_bt.get("brightness_temp"),
        "brightness_temp",
        rel_rms_tol=1e-5,
        rtol=1e-5,
        atol_scale=1e-5,
    )
    if use_ts_fluct:
        _assert_rel_close(
            gpu_bt.get("tau_21"),
            cpu_bt.get("tau_21"),
            "tau_21",
            rel_rms_tol=1e-5,
            rtol=1e-5,
            atol_scale=1e-5,
        )


@requires_cuda_runtime
@pytest.mark.filterwarnings(
    "ignore:^Your model .*uses the EPS conditional mass function:UserWarning"
)
@pytest.mark.parametrize(
    ("source_model", "use_ts", "use_metal", "use_mcgs", "extra_fields", "halo_median"),
    [
        ("E-INTEGRAL", True, False, False, False, False),
        ("CONST-ION-EFF", True, False, False, True, False),
        ("E-INTEGRAL", True, True, True, True, True),
    ],
)
def test_cuda_emissivity_fields_eulerian(
    tmp_path: Path,
    source_model: str,
    use_ts: bool,
    use_metal: bool,
    use_mcgs: bool,
    extra_fields: bool,
    halo_median: bool,
) -> None:
    """Compare CUDA and CPU ``ComputeEmissivityFields`` across Eulerian options."""
    inputs = _make_eulerian_inputs(
        random_seed=22,
        dim=48,
        hii_dim=16,
        box_len=64.0,
        n_threads=4 if use_mcgs else 1,
        min_redshift=8.0,
        max_redshift=20.0,
        source_model=source_model,
        use_ts_fluct=use_ts,
        use_metallicity=use_metal,
        use_mcgs=use_mcgs,
        use_reion_fb=use_mcgs,
        recomb_model="inhomogeneous" if use_mcgs else "none",
        halo_median=halo_median,
        power_spectrum="CLASS" if use_mcgs else "EH",
        v_cb_model="FLUCTS" if use_mcgs else "NONE",
    )
    prev_z = float(inputs.node_redshifts[-2])
    z = float(inputs.node_redshifts[-1])
    shape = (16, 16, 16)
    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "prep")):
        ic = p21.compute_initial_conditions(inputs=inputs, regenerate=True, write=False)
        pf = p21.perturb_field(
            redshift=z, inputs=inputs, initial_conditions=ic, write=False
        )

    prev_ts = None
    prev_ib = None
    if use_mcgs:
        rng = np.random.default_rng(77)
        prev_ts = p21.TsBox.new(inputs=inputs, redshift=prev_z)
        prev_ib = p21.IonizedBox.new(inputs=inputs, redshift=prev_z)
        prev_ts.set("J_21_LW", rng.uniform(0.0, 0.2, size=shape).astype(np.float32))
        prev_ib.set(
            "ionisation_rate_G12", rng.uniform(0.0, 0.5, size=shape).astype(np.float32)
        )
        prev_ib.set(
            "z_reion", np.where(rng.random(shape) > 0.8, 14.0, -1.0).astype(np.float32)
        )

    with p21.config.use(
        USE_CUDA=False,
        EXTRA_EMISSIVITY_FIELDS=extra_fields,
        direc=str(tmp_path / "cpu"),
    ):
        cpu_ef = p21.compute_emissivity_fields(
            redshift=z,
            inputs=inputs,
            initial_conditions=ic,
            perturbed_field=pf,
            previous_spin_temp=prev_ts,
            previous_ionize_box=prev_ib,
            regenerate=True,
            write=False,
        )
    with p21.config.use(
        USE_CUDA=True, EXTRA_EMISSIVITY_FIELDS=extra_fields, direc=str(tmp_path / "gpu")
    ):
        before = _get_stats().emissivity_runs
        gpu_ef = p21.compute_emissivity_fields(
            redshift=z,
            inputs=inputs,
            initial_conditions=ic,
            perturbed_field=pf,
            previous_spin_temp=prev_ts,
            previous_ionize_box=prev_ib,
            regenerate=True,
            write=False,
        )
        assert _get_stats().emissivity_runs == before + 1

    assert np.isclose(gpu_ef.log10_mturn_acg_ave, cpu_ef.log10_mturn_acg_ave, rtol=1e-6)
    assert np.isclose(gpu_ef.log10_mturn_mcg_ave, cpu_ef.log10_mturn_mcg_ave, rtol=1e-6)
    fields = ["n_ion", "sfrd_acg", "xray_emissivity"]
    if use_mcgs:
        fields.append("sfrd_mcg")
    if extra_fields:
        fields.extend(["halo_number", "halo_mass_density", "stellar_mass_density_acg"])
        if use_mcgs:
            fields.append("stellar_mass_density_mcg")
    for fname in fields:
        _assert_rel_close(
            gpu_ef.get(fname),
            cpu_ef.get(fname),
            f"emissivity.{fname}",
            rel_rms_tol=1e-4,
            rtol=1e-3,
            atol_scale=1e-4,
        )


@requires_cuda_runtime
@pytest.mark.parametrize(
    ("use_xray", "use_cmb", "use_lya_heat", "lya_ms", "use_mcgs", "non_cubic"),
    [
        (True, False, False, False, False, 1.0),
        (False, True, True, False, False, 1.0),
        (True, True, True, True, True, 1.5),
    ],
)
def test_cuda_radiation_and_spin_temp(
    tmp_path: Path,
    use_xray: bool,
    use_cmb: bool,
    use_lya_heat: bool,
    lya_ms: bool,
    use_mcgs: bool,
    non_cubic: float,
) -> None:
    """Compare multi-redshift radiation + spin temperature evolution on GPU vs CPU."""
    min_z, max_z, step, n_threads = (
        (12.0, 18.0, 1.2, 4) if use_mcgs else (10.0, 22.0, 1.18, 1)
    )
    inputs = _make_eulerian_inputs(
        random_seed=33,
        dim=48,
        hii_dim=16,
        box_len=64.0,
        non_cubic_factor=non_cubic,
        n_threads=n_threads,
        min_redshift=min_z,
        max_redshift=max_z,
        z_step_factor=step,
        use_ts_fluct=True,
        use_xray_heating=use_xray,
        use_cmb_heating=use_cmb,
        use_lya_heating=use_lya_heat,
        lya_multiple_scattering=lya_ms,
        use_mcgs=use_mcgs,
        use_reion_fb=use_mcgs,
        recomb_model="inhomogeneous" if use_mcgs else "none",
        power_spectrum="CLASS" if use_mcgs else "EH",
        v_cb_model="FLUCTS" if use_mcgs else "NONE",
    )
    redshifts = list(inputs.node_redshifts)  # already high-z -> low-z

    def _run_ts_chain(
        use_cuda: bool, tag: str
    ) -> tuple[p21.RadiationFields, p21.TsBox]:
        with p21.config.use(USE_CUDA=use_cuda, direc=str(tmp_path / tag)):
            ic = p21.compute_initial_conditions(
                inputs=inputs, regenerate=True, write=False
            )
            ef_hist: list[p21.EmissivityFields] = []
            prev_ts = None
            prev_ib = None
            prev_pf = None
            prev_rad_setup = None
            last_rad = None
            for z in redshifts:
                pf = p21.perturb_field(
                    redshift=z,
                    inputs=inputs,
                    initial_conditions=ic,
                    regenerate=True,
                    write=False,
                )
                ef = p21.compute_emissivity_fields(
                    redshift=z,
                    inputs=inputs,
                    initial_conditions=ic,
                    perturbed_field=pf,
                    previous_spin_temp=prev_ts,
                    previous_ionize_box=prev_ib,
                    regenerate=True,
                    write=False,
                )
                ef_hist.append(ef)
                if z <= inputs.simulation_options.Z_HEAT_MAX:
                    prev_rad_setup = setup_radiation_fields(
                        redshift=z,
                        emissivity_fields_list=ef_hist,
                        previous_rad_setup=prev_rad_setup,
                        previous_spin_temp=prev_ts,
                    )
                    rad = p21.compute_radiation_fields(
                        redshift=z,
                        emissivity_fields_list=ef_hist,
                        rad_setup=prev_rad_setup,
                        perturbed_field=pf,
                        previous_ionize_box=prev_ib,
                        previous_spin_temp=prev_ts,
                        regenerate=True,
                        write=False,
                    )
                    last_rad = rad
                else:
                    rad = None
                ts_box = p21.compute_spin_temperature(
                    initial_conditions=ic,
                    perturbed_field=pf,
                    inputs=inputs,
                    radiation_fields=rad,
                    previous_spin_temp=prev_ts,
                    regenerate=True,
                    write=False,
                )
                if use_mcgs or lya_ms:
                    prev_ib = p21.compute_ionization_field(
                        inputs=inputs,
                        initial_conditions=ic,
                        perturbed_field=pf,
                        previous_perturbed_field=prev_pf,
                        previous_ionized_box=prev_ib,
                        spin_temp=ts_box,
                        emissivity_fields=ef,
                        regenerate=True,
                        write=False,
                    )
                prev_ts = ts_box
                prev_pf = pf
            assert last_rad is not None
            assert prev_ts is not None
            return last_rad, prev_ts

    cpu_rad, cpu_ts = _run_ts_chain(False, "cpu")
    before = _get_stats()
    gpu_rad, gpu_ts = _run_ts_chain(True, "gpu")
    after = _get_stats()

    assert after.radiation_setup_runs > before.radiation_setup_runs
    assert after.radiation_update_runs > before.radiation_update_runs
    assert after.spin_temp_runs == before.spin_temp_runs + len(redshifts)

    for rname in gpu_rad.arrays:
        _assert_rel_close(
            gpu_rad.get(rname),
            cpu_rad.get(rname),
            f"radiation.{rname}",
            rel_rms_tol=5e-4,
            rtol=5e-3,
            atol_scale=5e-4,
        )
    for tname in ("spin_temperature", "kinetic_temp_neutral", "xray_ionised_fraction"):
        _assert_rel_close(
            gpu_ts.get(tname),
            cpu_ts.get(tname),
            f"ts.{tname}",
            rel_rms_tol=2e-4,
            rtol=2e-3,
            atol_scale=2e-4,
        )
    if use_mcgs:
        _assert_rel_close(
            gpu_ts.get("J_21_LW"),
            cpu_ts.get("J_21_LW"),
            "ts.J_21_LW",
            rel_rms_tol=5e-4,
            rtol=5e-3,
            atol_scale=5e-4,
        )


@requires_cuda_runtime
@pytest.mark.filterwarnings(
    "ignore:^Your model .*uses the EPS conditional mass function:UserWarning"
)
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("default", {}),
        ("const_eff", {"source_model": "CONST-ION-EFF"}),
        ("sharp_k", {"hii_filter": "sharp-k"}),
        ("gaussian", {"hii_filter": "gaussian"}),
        ("homo_recomb", {"recomb_model": "homogeneous", "cell_recomb": True}),
        ("inhomo_cell", {"recomb_model": "inhomogeneous", "cell_recomb": True}),
        ("inhomo_filt", {"recomb_model": "inhomogeneous", "cell_recomb": False}),
        ("sphere", {"ionise_entire_sphere": True, "recomb_model": "none"}),
        (
            "photoncons_z",
            {"photon_cons_type": "z-photoncons", "recomb_model": "inhomogeneous"},
        ),
        ("min_mem", {"recomb_model": "inhomogeneous", "minimize_memory": True}),
        ("noncubic", {"recomb_model": "inhomogeneous", "non_cubic_factor": 1.5}),
        (
            "mcgs_fb_ts",
            {
                "n_threads": 4,
                "use_ts_fluct": True,
                "use_mcgs": True,
                "use_reion_fb": True,
                "recomb_model": "inhomogeneous",
                "power_spectrum": "CLASS",
                "v_cb_model": "FLUCTS",
            },
        ),
    ],
)
def test_cuda_ionized_box_variants(
    tmp_path: Path, name: str, kwargs: dict[str, Any]
) -> None:
    """Compare CUDA and CPU ``ComputeIonizedBox`` across filters, recombinations, and flags."""
    max_z = 18.0 if name == "mcgs_fb_ts" else 20.0
    min_z = 10.0 if name == "mcgs_fb_ts" else 7.5
    inputs = _make_eulerian_inputs(
        random_seed=44,
        dim=48,
        hii_dim=16,
        box_len=64.0,
        min_redshift=min_z,
        max_redshift=max_z,
        z_step_factor=1.18,
        hii_eff_factor=45.0 if kwargs.get("source_model") == "CONST-ION-EFF" else None,
        **kwargs,
    )

    def _run_coeval_driver(use_cuda: bool, tag: str) -> p21.Coeval:
        if lib.photon_cons_allocated:
            lib.FreePhotonConsMemory()
        with p21.config.use(USE_CUDA=use_cuda, direc=str(tmp_path / tag)):
            return p21.run_coeval(
                inputs=inputs,
                out_redshifts=[float(inputs.node_redshifts[-1])],
                regenerate=True,
                write=False,
            )[0]

    cpu_cv = _run_coeval_driver(False, "cpu")
    before = _get_stats().ionization_runs
    gpu_cv = _run_coeval_driver(True, "gpu")
    after = _get_stats()
    assert after.ionization_runs > before
    assert abs(gpu_cv.ionized_box.global_xH - cpu_cv.ionized_box.global_xH) < 5e-3, name

    if not kwargs.get("minimize_memory", False):
        _assert_rel_close(
            gpu_cv.ionized_box.get("kinetic_temperature"),
            cpu_cv.ionized_box.get("kinetic_temperature"),
            f"{name}.kinetic_temperature",
            rel_rms_tol=5e-3,
            rtol=2e-2,
            atol_scale=5e-3,
        )
    if kwargs.get("recomb_model", "none") == "inhomogeneous":
        g_rec = gpu_cv.ionized_box.get("cumulative_recombinations")
        c_rec = cpu_cv.ionized_box.get("cumulative_recombinations")
        if np.max(np.abs(c_rec)) > 0:
            _assert_rel_close(
                g_rec,
                c_rec,
                f"{name}.cumulative_recombinations",
                rel_rms_tol=5e-3,
                rtol=2e-2,
                atol_scale=5e-3,
            )
    elif kwargs.get("recomb_model", "none") == "homogeneous":
        np.testing.assert_allclose(
            gpu_cv.ionized_box.get("cumulative_recombinations"),
            cpu_cv.ionized_box.get("cumulative_recombinations"),
            rtol=5e-3,
            atol=1e-4,
        )

    _assert_rel_close(
        gpu_cv.brightness_temp,
        cpu_cv.brightness_temp,
        f"{name}.brightness_temp",
        rel_rms_tol=2e-2,
        rtol=5e-2,
        atol_scale=2e-2,
    )


@requires_cuda_runtime
@pytest.mark.parametrize("use_exp_filter", [False, True])
def test_cuda_ionized_box_lagrangian_grids(
    tmp_path: Path, use_exp_filter: bool
) -> None:
    """Verify GPU ``ComputeIonizedBox`` on ``L-INTEGRAL`` with standard and exp-MFP filters."""
    inputs = _make_eulerian_inputs(
        random_seed=55,
        dim=48,
        hii_dim=16,
        box_len=64.0,
        min_redshift=8.0,
        max_redshift=16.0,
        z_step_factor=1.2,
        source_model="L-INTEGRAL",
        recomb_model="inhomogeneous",
        cell_recomb=use_exp_filter,
        use_exp_filter=use_exp_filter,
    )
    z = 8.0
    with p21.config.use(USE_CUDA=False, direc=str(tmp_path / "cpu")):
        cpu_cv = p21.run_coeval(
            inputs=inputs, out_redshifts=[z], regenerate=True, write=False
        )[0]
    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "gpu")):
        before = _get_stats().ionization_runs
        gpu_cv = p21.run_coeval(
            inputs=inputs, out_redshifts=[z], regenerate=True, write=False
        )[0]
        assert _get_stats().ionization_runs > before

    assert abs(gpu_cv.ionized_box.global_xH - cpu_cv.ionized_box.global_xH) < 5e-3
    assert np.isclose(
        gpu_cv.ionized_box.nion_unconditional_acg,
        cpu_cv.ionized_box.nion_unconditional_acg,
        rtol=1e-4,
    )


@requires_cuda_runtime
def test_cuda_global_evolution_fallback_hii1(tmp_path: Path) -> None:
    """Ensure ``HII_DIM=1`` global evolution keeps Phase 2 GPU counters at 0."""
    inputs = p21.InputParameters(random_seed=99).with_logspaced_redshifts(
        zmin=8.0, zmax=25.0, step=1.2
    )
    inputs = inputs.evolve_input_structs(
        Z_HEAT_MAX=25.0,
        SOURCE_MODEL="E-INTEGRAL",
        USE_TS_FLUCT=True,
        USE_EXP_FILTER=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
    )
    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "hii1")):
        lib.p21_cuda_reset_stats()
        ge = p21.run_global_evolution(inputs=inputs)
        assert np.all(np.isfinite(ge.quantities["brightness_temp"]))
        stats = _get_stats()
        assert stats.emissivity_runs == 0
        assert stats.radiation_setup_runs == 0
        assert stats.radiation_update_runs == 0
        assert stats.spin_temp_runs == 0
        assert stats.ionization_runs == 0
        assert stats.brightness_runs == 0


@requires_cuda_runtime
@pytest.mark.parametrize("use_mcgs", [False, True])
def test_cuda_coeval_lightcone_residency_and_cleanup(
    tmp_path: Path, use_mcgs: bool
) -> None:
    """Test end-to-end Coeval + LightCone parity, device residency hits, and clean teardown."""
    min_z, max_z, step, n_threads = (
        (10.0, 18.0, 1.2, 4) if use_mcgs else (7.5, 22.0, 1.15, 1)
    )
    inputs = _make_eulerian_inputs(
        random_seed=777,
        dim=48,
        hii_dim=16,
        box_len=64.0,
        n_threads=n_threads,
        min_redshift=min_z,
        max_redshift=max_z,
        z_step_factor=step,
        use_ts_fluct=True,
        use_mcgs=use_mcgs,
        use_reion_fb=use_mcgs,
        recomb_model="inhomogeneous",
        power_spectrum="CLASS" if use_mcgs else "EH",
        v_cb_model="FLUCTS" if use_mcgs else "NONE",
    )
    lightconer = p21.RectilinearLightconer.between_redshifts(
        min_redshift=min_z + 0.5,
        max_redshift=max_z - 1.5,
        quantities=("brightness_temp", "neutral_fraction", "spin_temperature"),
        resolution=inputs.simulation_options.cell_size,
    )

    with p21.config.use(USE_CUDA=False, direc=str(tmp_path / "cpu")):
        cpu_lc = p21.run_lightcone(
            inputs=inputs, lightconer=lightconer, regenerate=True, write=False
        )

    lib.p21_cuda_reset_stats()
    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "gpu")):
        gpu_lc = p21.run_lightcone(
            inputs=inputs, lightconer=lightconer, regenerate=True, write=False
        )
    stats = _get_stats()

    assert "cupy" not in sys.modules
    assert stats.ic_runs == 1
    assert stats.perturb_runs == len(inputs.node_redshifts)
    assert stats.emissivity_runs == len(inputs.node_redshifts)
    assert stats.radiation_setup_runs > 0
    assert stats.radiation_update_runs > 0
    assert stats.spin_temp_runs == len(inputs.node_redshifts)
    assert stats.ionization_runs == len(inputs.node_redshifts)
    assert stats.brightness_runs == len(inputs.node_redshifts)
    assert stats.registry_hits > 0

    for gkey in ("neutral_fraction", "spin_temperature", "brightness_temp"):
        np.testing.assert_allclose(
            gpu_lc.global_quantities[gkey],
            cpu_lc.global_quantities[gkey],
            rtol=2e-3,
            atol=2e-3,
            err_msg=f"global.{gkey}",
        )

    ps_gpu = get_power(
        gpu_lc.lightcones["brightness_temp"],
        boxlength=gpu_lc.lightcone_dimensions,
        bins_upto_boxlen=True,
    )
    ps_cpu = get_power(
        cpu_lc.lightcones["brightness_temp"],
        boxlength=cpu_lc.lightcone_dimensions,
        bins_upto_boxlen=True,
    )
    valid = np.isfinite(ps_gpu.power) & np.isfinite(ps_cpu.power)
    assert np.any(valid)
    np.testing.assert_allclose(
        ps_gpu.power[valid], ps_cpu.power[valid], rtol=2e-3, atol=1e-4
    )

    # Verify clean registry/pool teardown and restart.
    lib.p21_cuda_registry_clear()
    lib.p21_cuda_cleanup()
    gc.collect()
    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "gpu_restart")):
        lib.p21_cuda_reset_stats()
        cv_restart = p21.run_coeval(
            inputs=inputs,
            out_redshifts=[float(inputs.node_redshifts[-1])],
            regenerate=True,
            write=False,
        )[0]
        assert np.all(np.isfinite(cv_restart.brightness_temp))
        assert _get_stats().brightness_runs >= 1
