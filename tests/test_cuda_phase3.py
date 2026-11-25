"""Phase 3 tests for the native C/CUDA Lagrangian & Halo pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from powerbox import get_power

import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.wrapper import cfuncs as cf
from py21cmfast.wrapper.exceptions import ArgumentValueError

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def _get_stats() -> Any:
    """Return a snapshot of the backend ``P21CudaStats`` struct."""
    stats = ffi.new("P21CudaStats *")
    lib.p21_cuda_get_stats(stats)
    return stats


def _make_inputs(
    *,
    random_seed: int = 30301,
    dim: int = 48,
    hii_dim: int = 16,
    box_len: float = 64.0,
    non_cubic_factor: float = 1.0,
    n_threads: int = 1,
    node_redshifts: list[float] | tuple[float, ...] = (12.0, 10.0),
    source_model: str = "L-INTEGRAL",
    hmf: str = "ST",
    perturb_algorithm: str = "2LPT",
    perturb_on_high_res: bool = False,
    use_ts_fluct: bool = True,
    use_mcgs: bool = False,
    use_reion_fb: bool = False,
    use_metallicity: bool = False,
    use_upper_turnover: bool = False,
    recomb_model: str = "inhomogeneous",
    dexm_optimize: bool = True,
    sample_method: str = "NUMBER-LIMITED",
    sampler_min_mass: float = 1e8,
) -> p21.InputParameters:
    """Construct validated ``InputParameters`` for Phase 3 Lagrangian/Halo tests."""
    extra: dict[str, Any] = {}
    if use_mcgs:
        extra["M_TURN_STELLAR_FEEDBACK"] = 5.0
        extra["V_CB_MODEL"] = "FLUCTS"
        extra["POWER_SPECTRUM"] = "CLASS"
        extra["K_MAX_FOR_CLASS"] = 1.0
    if recomb_model != "none":
        extra["R_BUBBLE_MAX"] = 50.0 if box_len >= 50.0 else box_len
    extra["USE_UPPER_STELLAR_TURNOVER"] = (
        False if source_model == "L-INTEGRAL" else use_upper_turnover
    )

    return p21.InputParameters(
        random_seed=random_seed,
        node_redshifts=list(node_redshifts),
    ).evolve_input_structs(
        DIM=dim,
        HII_DIM=hii_dim,
        BOX_LEN=box_len,
        NON_CUBIC_FACTOR=non_cubic_factor,
        N_THREADS=n_threads,
        Z_HEAT_MAX=float(max(node_redshifts)),
        SOURCE_MODEL=source_model,
        HMF=hmf,
        PERTURB_ALGORITHM=perturb_algorithm,
        PERTURB_ON_HIGH_RES=perturb_on_high_res,
        USE_TS_FLUCT=use_ts_fluct,
        USE_MCGS=use_mcgs,
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=use_reion_fb,
        USE_METALLICITY=use_metallicity,
        RECOMB_MODEL=recomb_model,
        DEXM_OPTIMIZE=dexm_optimize,
        SAMPLE_METHOD=sample_method,
        SAMPLER_MIN_MASS=sampler_min_mass,
        **extra,
    )


def _assert_field_close(
    cpu_arr: np.ndarray,
    gpu_arr: np.ndarray,
    *,
    name: str,
    rel_rms_tol: float = 2e-4,
    rtol: float = 5e-4,
    atol: float = 1e-6,
) -> None:
    """Check finite arrays, relative RMS, and elementwise closeness."""
    x = np.asarray(cpu_arr, dtype=np.float64)
    y = np.asarray(gpu_arr, dtype=np.float64)
    assert x.shape == y.shape, f"{name}: shape {x.shape} != {y.shape}"
    assert np.all(np.isfinite(x)), f"{name}: non-finite CPU values"
    assert np.all(np.isfinite(y)), f"{name}: non-finite GPU values"
    cpu_rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    diff_rms = float(np.sqrt(np.mean((y - x) ** 2))) if x.size else 0.0
    rel_rms = diff_rms / cpu_rms if cpu_rms > 0.0 else diff_rms
    assert rel_rms <= rel_rms_tol, f"{name}: rel_rms={rel_rms:.3e} > {rel_rms_tol:.3e}"
    np.testing.assert_allclose(y, x, rtol=rtol, atol=atol, err_msg=name)


def test_phase3_config_and_stats_api() -> None:
    """Verify Phase 3 config toggle and extended ``P21CudaStats`` ABI exist in all builds."""
    assert "CUDA_FAST_HALO_SAMPLER" in p21.config
    orig = p21.config["CUDA_FAST_HALO_SAMPLER"]
    try:
        with p21.config.use(CUDA_FAST_HALO_SAMPLER=True):
            assert lib.p21_cuda_get_fast_halo_sampler() == 1
        with p21.config.use(CUDA_FAST_HALO_SAMPLER=False):
            assert lib.p21_cuda_get_fast_halo_sampler() == 0
    finally:
        assert p21.config["CUDA_FAST_HALO_SAMPLER"] == orig

    stats = _get_stats()
    for field in (
        "halo_dexm_runs",
        "halo_sampler_runs",
        "halo_perturb_runs",
        "dndm_table_runs",
    ):
        assert isinstance(getattr(stats, field), int)
        assert getattr(stats, field) >= 0
    assert "cupy" not in sys.modules


@requires_cuda_runtime
@pytest.mark.filterwarnings(
    "ignore:A selection of a mass function other than Press-Schechter.*:UserWarning"
)
@pytest.mark.parametrize(
    "case",
    [
        "lowres_base",
        "lowres_mcgs_fb_met",
        "hires_met",
        "watson_fixmean_noncubic",
    ],
)
def test_lagrangian_emissivity_l_integral(tmp_path: Path, case: str) -> None:
    """Test GPU ``L-INTEGRAL`` ``ComputeEmissivityFields`` parity across variants."""
    if case == "lowres_base":
        inputs = _make_inputs(
            random_seed=1101,
            perturb_on_high_res=False,
            use_ts_fluct=True,
            use_mcgs=False,
            use_reion_fb=False,
            use_metallicity=False,
            non_cubic_factor=1.0,
            hmf="ST",
        )
    elif case == "lowres_mcgs_fb_met":
        inputs = _make_inputs(
            random_seed=1102,
            perturb_on_high_res=False,
            use_ts_fluct=True,
            use_mcgs=True,
            use_reion_fb=True,
            use_metallicity=True,
            non_cubic_factor=1.0,
            hmf="ST",
        )
    elif case == "hires_met":
        inputs = _make_inputs(
            random_seed=1104,
            perturb_on_high_res=True,
            use_ts_fluct=True,
            use_mcgs=False,
            use_reion_fb=False,
            use_metallicity=True,
            non_cubic_factor=1.0,
            hmf="ST",
        )
    else:
        inputs = _make_inputs(
            random_seed=1103,
            dim=40,
            hii_dim=20,
            box_len=50.0,
            perturb_on_high_res=False,
            use_ts_fluct=True,
            use_mcgs=False,
            use_reion_fb=False,
            use_metallicity=True,
            non_cubic_factor=1.2,
            hmf="WATSON",
        )

    redshift = 10.0

    def _run_case(use_cuda: bool, subdir: str):
        rng = np.random.default_rng(9100)
        with p21.config.use(
            USE_CUDA=use_cuda,
            EXTRA_EMISSIVITY_FIELDS=True,
            direc=str(tmp_path / subdir),
        ):
            ics = p21.compute_initial_conditions(
                inputs=inputs, regenerate=True, write=False
            )
            pf = p21.perturb_field(
                redshift=redshift, initial_conditions=ics, regenerate=True, write=False
            )
            prev_ts = p21.TsBox.new(inputs=inputs, redshift=12.0)
            prev_ts._init_arrays()
            for name, arr in prev_ts.arrays.items():
                if name == "J_21_LW":
                    val = rng.uniform(0.01, 2.0, size=arr.shape).astype(arr.dtype)
                elif name in ("spin_temperature", "kinetic_temp_neutral"):
                    val = np.full(arr.shape, 25.0, dtype=arr.dtype)
                elif name == "xray_ionised_fraction":
                    val = np.full(arr.shape, 0.01, dtype=arr.dtype)
                else:
                    val = np.zeros(arr.shape, dtype=arr.dtype)
                prev_ts.set(name, val)

            prev_ion = p21.IonizedBox.new(inputs=inputs, redshift=12.0)
            prev_ion._init_arrays()
            for name, arr in prev_ion.arrays.items():
                if name == "ionisation_rate_G12":
                    val = rng.uniform(0.05, 0.8, size=arr.shape).astype(arr.dtype)
                elif name == "z_reion":
                    val = rng.uniform(9.5, 13.0, size=arr.shape).astype(arr.dtype)
                elif name == "neutral_fraction":
                    val = np.full(arr.shape, 0.95, dtype=arr.dtype)
                else:
                    val = np.zeros(arr.shape, dtype=arr.dtype)
                prev_ion.set(name, val)

            lib.p21_cuda_reset_stats()
            ef = p21.compute_emissivity_fields(
                redshift=redshift,
                inputs=inputs,
                initial_conditions=ics,
                perturbed_field=pf,
                previous_spin_temp=prev_ts,
                previous_ionize_box=prev_ion,
                regenerate=True,
                write=False,
            )
            ib = None
            if case == "lowres_mcgs_fb_met":
                pf_prev = p21.perturb_field(
                    redshift=12.0, initial_conditions=ics, regenerate=True, write=False
                )
                curr_ts = p21.TsBox.new(inputs=inputs, redshift=redshift)
                curr_ts._init_arrays()
                for name in curr_ts.arrays:
                    curr_ts.set(name, prev_ts.get(name))
                ib = p21.compute_ionization_field(
                    inputs=inputs,
                    initial_conditions=ics,
                    perturbed_field=pf,
                    previous_perturbed_field=pf_prev,
                    previous_ionized_box=prev_ion,
                    spin_temp=curr_ts,
                    emissivity_fields=ef,
                    regenerate=True,
                    write=False,
                )
            return ef, ib, _get_stats()

    ef_cpu, ib_cpu, _ = _run_case(False, "cpu")
    ef_gpu, ib_gpu, stats_gpu = _run_case(True, "gpu")

    assert "cupy" not in sys.modules
    assert stats_gpu.emissivity_runs == 1
    assert np.isclose(
        ef_gpu.log10_mturn_acg_ave, ef_cpu.log10_mturn_acg_ave, rtol=1e-5, atol=1e-6
    )
    assert np.isclose(
        ef_gpu.log10_mturn_mcg_ave, ef_cpu.log10_mturn_mcg_ave, rtol=1e-5, atol=1e-6
    )

    for key in (
        "n_ion",
        "sfrd_acg",
        "xray_emissivity",
        "fesc_weighted_sfrd",
        "halo_number",
        "halo_mass_density",
        "stellar_mass_density_acg",
    ):
        if ef_cpu.has(key):
            _assert_field_close(
                ef_cpu.get(key), ef_gpu.get(key), name=key, rel_rms_tol=5e-4, rtol=2e-3
            )
    if inputs.astro_options.USE_MCGS:
        for key in ("sfrd_mcg", "stellar_mass_density_mcg"):
            _assert_field_close(
                ef_cpu.get(key), ef_gpu.get(key), name=key, rel_rms_tol=5e-4, rtol=2e-3
            )
    if ib_cpu is not None and ib_gpu is not None:
        _assert_field_close(
            ib_cpu.get("neutral_fraction"),
            ib_gpu.get("neutral_fraction"),
            name="neutral_fraction",
            rel_rms_tol=5e-4,
            rtol=5e-3,
            atol=1e-5,
        )


@requires_cuda_runtime
@pytest.mark.parametrize(
    ("perturb_alg", "use_mcgs", "use_reion_fb"),
    [
        ("2LPT", True, True),
        ("ZELDOVICH", False, False),
    ],
)
def test_perturbed_halo_catalog_parity(
    tmp_path: Path, perturb_alg: str, use_mcgs: bool, use_reion_fb: bool
) -> None:
    """Test GPU ``ComputePerturbedHaloCatalog`` parity including zero-mass halo skip."""
    inputs = _make_inputs(
        random_seed=2201,
        dim=48,
        hii_dim=16,
        box_len=64.0,
        node_redshifts=(12.0, 10.0),
        source_model="CHMF-SAMPLER",
        perturb_algorithm=perturb_alg,
        perturb_on_high_res=True,
        use_ts_fluct=True,
        use_mcgs=use_mcgs,
        use_reion_fb=use_reion_fb,
        use_metallicity=True,
        use_upper_turnover=True,
        recomb_model="inhomogeneous",
        sampler_min_mass=2e9,
    )

    with p21.config.use(USE_CUDA=False, direc=str(tmp_path / "prep")):
        ics = p21.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
        halos = p21.determine_halo_catalog(
            redshift=10.0,
            initial_conditions=ics,
            inputs=inputs,
            regenerate=True,
            write=False,
        )
        assert halos.n_halos > 5
        halos.trim_to_n_halos()
        mod_masses = halos.get("halo_masses").copy()
        mod_masses[0] = 0.0
        halos.set("halo_masses", mod_masses)

        rng = np.random.default_rng(77)
        prev_ts = p21.TsBox.new(inputs=inputs, redshift=12.0)
        prev_ts._init_arrays()
        for name, arr in prev_ts.arrays.items():
            val = (
                rng.uniform(0.05, 1.5, size=arr.shape).astype(arr.dtype)
                if name == "J_21_LW"
                else np.zeros(arr.shape, dtype=arr.dtype)
            )
            prev_ts.set(name, val)

        prev_ion = p21.IonizedBox.new(inputs=inputs, redshift=12.0)
        prev_ion._init_arrays()
        for name, arr in prev_ion.arrays.items():
            if name == "ionisation_rate_G12":
                val = rng.uniform(0.05, 0.7, size=arr.shape).astype(arr.dtype)
            elif name == "z_reion":
                val = rng.uniform(9.5, 13.0, size=arr.shape).astype(arr.dtype)
            else:
                val = np.zeros(arr.shape, dtype=arr.dtype)
            prev_ion.set(name, val)

        cpu_pt = p21.perturb_halo_catalog(
            initial_conditions=ics,
            halo_catalog=halos,
            previous_spin_temp=prev_ts,
            previous_ionize_box=prev_ion,
            regenerate=True,
            write=False,
        )

    lib.p21_cuda_reset_stats()
    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "gpu")):
        gpu_pt = p21.perturb_halo_catalog(
            initial_conditions=ics,
            halo_catalog=halos,
            previous_spin_temp=prev_ts,
            previous_ionize_box=prev_ion,
            regenerate=True,
            write=False,
        )
    stats = _get_stats()

    assert "cupy" not in sys.modules
    assert stats.halo_perturb_runs == 1
    assert gpu_pt.n_halos == cpu_pt.n_halos == halos.n_halos
    assert gpu_pt.get("halo_masses")[0] == 0.0

    for key in (
        "halo_coords",
        "halo_masses",
        "stellar_masses_acg",
        "sfr_acg",
        "n_ion",
        "xray_luminosity",
        "fesc_weighted_sfr",
    ):
        np.testing.assert_allclose(
            gpu_pt.get(key),
            cpu_pt.get(key),
            rtol=1e-4,
            atol=1e-6,
            err_msg=key,
        )
    if use_mcgs:
        for key in ("stellar_masses_mcg", "sfr_mcg"):
            np.testing.assert_allclose(
                gpu_pt.get(key),
                cpu_pt.get(key),
                rtol=1e-4,
                atol=1e-6,
                err_msg=key,
            )


@requires_cuda_runtime
@pytest.mark.filterwarnings(
    "ignore:You have turned off caching for the perturbed halo fields.*:UserWarning",
    "ignore:Trying to purge array .*:UserWarning",
)
@pytest.mark.parametrize(
    ("dexm_optimize", "non_cubic_factor"),
    [
        (True, 1.0),
        (False, 1.0),
        (True, 1.25),
    ],
)
def test_dexm_esf_parity(
    tmp_path: Path, dexm_optimize: bool, non_cubic_factor: float
) -> None:
    """Test GPU ``DEXM-ESF`` halo finder + emissivity + brightness temp parity vs CPU."""
    inputs = _make_inputs(
        random_seed=3301,
        dim=48,
        hii_dim=16,
        box_len=48.0,
        non_cubic_factor=non_cubic_factor,
        node_redshifts=(10.0, 8.0),
        source_model="DEXM-ESF",
        dexm_optimize=dexm_optimize,
        use_ts_fluct=False,
        recomb_model="none",
        use_upper_turnover=True,
    ).evolve_input_structs(DEXM_OPTIMIZE_MINMASS=1e9)

    with p21.config.use(USE_CUDA=False, direc=str(tmp_path / "cpu")):
        cpu_ics = p21.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
        cpu_halos = p21.determine_halo_catalog(
            redshift=8.0,
            initial_conditions=cpu_ics,
            inputs=inputs,
            regenerate=True,
            write=False,
        )
        cpu_cv = p21.run_coeval(
            inputs=inputs,
            out_redshifts=(8.0,),
            initial_conditions=cpu_ics,
            regenerate=True,
            write=False,
        )[0]

    lib.p21_cuda_reset_stats()
    with p21.config.use(USE_CUDA=True, direc=str(tmp_path / "gpu")):
        gpu_ics = p21.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
        gpu_halos = p21.determine_halo_catalog(
            redshift=8.0,
            initial_conditions=gpu_ics,
            inputs=inputs,
            regenerate=True,
            write=False,
        )
        gpu_cv = p21.run_coeval(
            inputs=inputs,
            out_redshifts=(8.0,),
            initial_conditions=gpu_ics,
            regenerate=True,
            write=False,
        )[0]
    stats = _get_stats()

    assert "cupy" not in sys.modules
    assert stats.halo_dexm_runs >= 1
    assert gpu_halos.n_halos > 0
    # Slight single-precision cuFFT differences near threshold can shift at most ~1% of borderline halos.
    assert abs(gpu_halos.n_halos - cpu_halos.n_halos) <= max(
        2, int(0.02 * cpu_halos.n_halos)
    )
    if gpu_halos.n_halos == cpu_halos.n_halos:
        np.testing.assert_allclose(
            gpu_halos.get("halo_masses"), cpu_halos.get("halo_masses"), rtol=1e-4
        )
        np.testing.assert_allclose(
            gpu_halos.get("halo_coords"), cpu_halos.get("halo_coords"), rtol=1e-4
        )

    _assert_field_close(
        cpu_cv.brightness_temp,
        gpu_cv.brightness_temp,
        name="dexm_brightness_temp",
        rel_rms_tol=5e-3,
        rtol=5e-2,
        atol=5e-1,
    )


@requires_cuda_runtime
@pytest.mark.parametrize("hmf", ["ST", "PS"])
def test_dndm_tables_parity(hmf: str) -> None:
    """Test GPU ``initialize_dndm_tables`` & ``initialize_dndm_inverse_table`` parity vs CPU."""
    redshift = 10.0
    z_prev = 8.5
    inputs = _make_inputs(
        random_seed=4401,
        source_model="CHMF-SAMPLER",
        hmf=hmf,
        node_redshifts=(12.0, redshift, z_prev),
        sampler_min_mass=1e8,
        use_ts_fluct=False,
        recomb_model="none",
    ).evolve_input_structs(USE_INTERPOLATION_TABLES="hmf-interpolation")

    deltas = np.linspace(-0.9, 1.35, 24)
    masses = np.geomspace(1e9, 1e12, 20)
    probs = np.linspace(0.05, 0.95, len(deltas))

    for cond_type, z_desc, cond_arr in (
        ("grid", None, deltas),
        ("cat", z_prev, masses),
    ):
        with p21.config.use(USE_CUDA=False):
            n_cpu, m_cpu = cf.evaluate_condition_integrals(
                inputs=inputs,
                redshift=redshift,
                redshift_prev=z_desc,
                cond_array=cond_arr,
            )
            p_arr = probs[: len(cond_arr)]
            inv_cpu = cf.evaluate_inverse_table(
                inputs=inputs,
                cond_array=cond_arr,
                probabilities=p_arr,
                redshift=redshift,
                redshift_prev=z_desc,
            )

        lib.p21_cuda_reset_stats()
        with p21.config.use(USE_CUDA=True):
            n_gpu, m_gpu = cf.evaluate_condition_integrals(
                inputs=inputs,
                redshift=redshift,
                redshift_prev=z_desc,
                cond_array=cond_arr,
            )
            inv_gpu = cf.evaluate_inverse_table(
                inputs=inputs,
                cond_array=cond_arr,
                probabilities=p_arr,
                redshift=redshift,
                redshift_prev=z_desc,
            )
        stats = _get_stats()
        assert (
            stats.dndm_table_runs >= 2
        ), f"{hmf}/{cond_type}: dndm_table_runs={stats.dndm_table_runs}"

        np.testing.assert_allclose(
            n_gpu, n_cpu, rtol=1e-3, atol=1e-12, err_msg=f"{hmf}/{cond_type} nhalo"
        )
        np.testing.assert_allclose(
            m_gpu, m_cpu, rtol=1e-3, atol=1e-6, err_msg=f"{hmf}/{cond_type} mcoll"
        )
        np.testing.assert_allclose(
            inv_gpu, inv_cpu, rtol=2e-3, atol=1e-6, err_msg=f"{hmf}/{cond_type} inv"
        )


@requires_cuda_runtime
@pytest.mark.filterwarnings(
    "ignore:You are setting R_BUBBLE_MAX != 50.*:UserWarning",
    "ignore:You have turned off caching for the perturbed halo fields.*:UserWarning",
    "ignore:Trying to purge array .*:UserWarning",
)
def test_chmf_default_lightcone_parity(tmp_path: Path) -> None:
    """Test multi-node ``CHMF-SAMPLER`` lightcone parity under default ``CUDA_FAST_HALO_SAMPLER=False``."""
    nodes = [11.5, 11.0, 10.5, 10.0, 9.5]
    inputs = _make_inputs(
        random_seed=81033,
        dim=48,
        hii_dim=16,
        box_len=48.0,
        n_threads=1,
        node_redshifts=nodes,
        source_model="CHMF-SAMPLER",
        use_ts_fluct=True,
        recomb_model="inhomogeneous",
        use_metallicity=True,
        use_upper_turnover=True,
        sampler_min_mass=2e9,
    )
    lc = p21.RectilinearLightconer.between_redshifts(
        min_redshift=9.6,
        max_redshift=11.3,
        quantities=("brightness_temp", "neutral_fraction", "spin_temperature"),
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )

    with p21.config.use(
        USE_CUDA=False, CUDA_FAST_HALO_SAMPLER=False, direc=str(tmp_path / "cpu")
    ):
        cpu_lc = p21.run_lightcone(
            inputs=inputs, lightconer=lc, regenerate=True, write=False
        )

    lib.p21_cuda_reset_stats()
    with p21.config.use(
        USE_CUDA=True, CUDA_FAST_HALO_SAMPLER=False, direc=str(tmp_path / "gpu")
    ):
        gpu_lc = p21.run_lightcone(
            inputs=inputs, lightconer=lc, regenerate=True, write=False
        )
    stats = _get_stats()

    assert "cupy" not in sys.modules
    assert stats.dndm_table_runs > 0
    assert stats.halo_dexm_runs >= 1
    assert stats.emissivity_runs == len(nodes)
    assert stats.halo_sampler_runs == 0

    for gkey in ("brightness_temp", "neutral_fraction", "spin_temperature"):
        np.testing.assert_allclose(
            gpu_lc.global_quantities[gkey],
            cpu_lc.global_quantities[gkey],
            rtol=2e-3,
            atol=5e-2,
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
    np.testing.assert_allclose(
        ps_gpu.power[valid], ps_cpu.power[valid], rtol=5e-3, atol=1e-2
    )


@requires_cuda_runtime
@pytest.mark.parametrize(
    "sample_method",
    ["NUMBER-LIMITED", "MASS-LIMITED", "PARTITION", "BINARY-SPLIT"],
)
def test_fast_halo_sampler_methods_and_stats(
    tmp_path: Path, sample_method: str
) -> None:
    """Test opt-in ``CUDA_FAST_HALO_SAMPLER=True`` across all 4 ``SAMPLE_METHOD`` values."""
    inputs = _make_inputs(
        random_seed=5501,
        dim=48,
        hii_dim=16,
        box_len=64.0,
        n_threads=2,
        node_redshifts=(11.5, 10.5),
        source_model="CHMF-SAMPLER",
        hmf="ST",
        sample_method=sample_method,
        sampler_min_mass=5e8,
        use_ts_fluct=False,
        recomb_model="none",
    )

    with p21.config.use(
        USE_CUDA=False, CUDA_FAST_HALO_SAMPLER=False, direc=str(tmp_path / "cpu")
    ):
        cpu_ics = p21.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
        cpu_desc = p21.determine_halo_catalog(
            redshift=10.5,
            initial_conditions=cpu_ics,
            inputs=inputs,
            regenerate=True,
            write=False,
        )
        # Upstream CPU PARTITION crashes in Stochasticity.c::stoc_partition_sample during progenitor
        # sampling (`USE_CUDA=False`); compare against CPU progenitors for the other 3 methods.
        cpu_prog = None
        if sample_method != "PARTITION":
            cpu_prog = p21.determine_halo_catalog(
                redshift=11.5,
                initial_conditions=cpu_ics,
                descendant_halos=cpu_desc,
                inputs=inputs,
                regenerate=True,
                write=False,
            )

    def _run_fast(subdir: str):
        lib.p21_cuda_reset_stats()
        with p21.config.use(
            USE_CUDA=True, CUDA_FAST_HALO_SAMPLER=True, direc=str(tmp_path / subdir)
        ):
            ics = p21.compute_initial_conditions(
                inputs=inputs, regenerate=True, write=False
            )
            desc = p21.determine_halo_catalog(
                redshift=10.5,
                initial_conditions=ics,
                inputs=inputs,
                regenerate=True,
                write=False,
            )
            prog = p21.determine_halo_catalog(
                redshift=11.5,
                initial_conditions=ics,
                descendant_halos=desc,
                inputs=inputs,
                regenerate=True,
                write=False,
            )
            return desc, prog, _get_stats()

    g1_desc, g1_prog, stats1 = _run_fast("gpu1")
    g2_desc, g2_prog, _ = _run_fast("gpu2")

    assert "cupy" not in sys.modules
    assert stats1.halo_sampler_runs == 2

    # Same seed on GPU must be deterministically reproducible.
    assert g1_desc.n_halos == g2_desc.n_halos
    assert g1_prog.n_halos == g2_prog.n_halos
    np.testing.assert_array_equal(
        g1_desc.get("halo_masses"), g2_desc.get("halo_masses")
    )
    np.testing.assert_array_equal(
        g1_desc.get("halo_coords"), g2_desc.get("halo_coords")
    )
    np.testing.assert_array_equal(
        g1_prog.get("halo_masses"), g2_prog.get("halo_masses")
    )
    np.testing.assert_array_equal(
        g1_prog.get("halo_coords"), g2_prog.get("halo_coords")
    )

    # Statistical consistency against CPU GSL reference on descendant samples.
    assert cpu_desc.n_halos > 50
    assert g1_desc.n_halos > 50
    assert abs(g1_desc.n_halos - cpu_desc.n_halos) / cpu_desc.n_halos < 0.25
    assert (
        abs(
            float(g1_desc.get("halo_masses").sum())
            - float(cpu_desc.get("halo_masses").sum())
        )
        / float(cpu_desc.get("halo_masses").sum())
        < 0.25
    )

    if sample_method == "PARTITION":
        assert g1_prog.n_halos > 20
        prog_masses = g1_prog.get("halo_masses")
        desc_masses = g1_desc.get("halo_masses")
        assert np.all(
            prog_masses >= inputs.simulation_options.SAMPLER_MIN_MASS * (1.0 - 1e-6)
        )
        with p21.config.use(USE_CUDA=True):
            _, exp_mcoll = cf.evaluate_condition_integrals(
                inputs=inputs,
                redshift=11.5,
                redshift_prev=10.5,
                cond_array=desc_masses.astype(np.float64),
            )
        prog_mass_sum = float(prog_masses.sum())
        desc_mass_sum = float(desc_masses.sum())
        exp_mass_sum = float(np.sum(exp_mcoll))
        assert exp_mass_sum > 0.0
        assert abs(prog_mass_sum - exp_mass_sum) / exp_mass_sum < 0.35
        assert prog_mass_sum <= 1.05 * desc_mass_sum
    else:
        assert cpu_prog is not None
        assert cpu_prog.n_halos > 20
        assert g1_prog.n_halos > 20
        assert abs(g1_prog.n_halos - cpu_prog.n_halos) / cpu_prog.n_halos < 0.30
        assert (
            abs(
                float(g1_prog.get("halo_masses").sum())
                - float(cpu_prog.get("halo_masses").sum())
            )
            / float(cpu_prog.get("halo_masses").sum())
            < 0.30
        )


@requires_cuda_runtime
def test_fast_halo_sampler_overflow_error(tmp_path: Path) -> None:
    """Verify opt-in GPU halo sampler reproduces upstream overflow diagnostics and error type."""
    inputs = _make_inputs(
        random_seed=6601,
        dim=48,
        hii_dim=16,
        box_len=64.0,
        n_threads=4,
        node_redshifts=(8.0, 6.0),
        source_model="CHMF-SAMPLER",
        sampler_min_mass=2e8,
        use_ts_fluct=False,
        recomb_model="none",
    )

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as stderr_file:
        stderr_path = Path(stderr_file.name)

    try:
        old_stderr = os.dup(2)
        with stderr_path.open("w") as f:
            os.dup2(f.fileno(), 2)
        try:
            with p21.config.use(
                USE_CUDA=True,
                CUDA_FAST_HALO_SAMPLER=True,
                HALO_CATALOG_MEM_FACTOR=0.6,
                direc=str(tmp_path / "ovf"),
            ):
                ics = p21.compute_initial_conditions(
                    inputs=inputs, regenerate=True, write=False
                )
                with pytest.raises(ArgumentValueError):
                    p21.determine_halo_catalog(
                        redshift=6.0,
                        initial_conditions=ics,
                        inputs=inputs,
                        regenerate=True,
                        write=False,
                    )
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)

        stderr_text = stderr_path.read_text()
        assert "Halo buffer overflow (allocated " in stderr_text
        for t in range(4):
            assert f"Thread {t}:" in stderr_text
    finally:
        stderr_path.unlink(missing_ok=True)


@requires_cuda_runtime
def test_fast_halo_sampler_stream_independence(tmp_path: Path) -> None:
    """Verify fast GPU halo sampler produces independent property RNG streams."""
    inputs = _make_inputs(
        random_seed=7701,
        dim=48,
        hii_dim=16,
        box_len=64.0,
        n_threads=4,
        node_redshifts=(11.0, 10.0),
        source_model="CHMF-SAMPLER",
        hmf="ST",
        sample_method="MASS-LIMITED",
        sampler_min_mass=2e8,
        use_ts_fluct=False,
        recomb_model="none",
    )

    with p21.config.use(
        USE_CUDA=True,
        CUDA_FAST_HALO_SAMPLER=True,
        HALO_CATALOG_MEM_FACTOR=2.0,
        direc=str(tmp_path / "indep"),
    ):
        ics = p21.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
        desc = p21.determine_halo_catalog(
            redshift=10.0,
            initial_conditions=ics,
            inputs=inputs,
            regenerate=True,
            write=False,
        )
        prog = p21.determine_halo_catalog(
            redshift=11.0,
            initial_conditions=ics,
            descendant_halos=desc,
            inputs=inputs,
            regenerate=True,
            write=False,
        )

    for label, hc in (("desc", desc), ("prog", prog)):
        n = int(hc.n_halos)
        assert n > 1000, f"{label}: too few halos ({n})"
        sr = np.asarray(hc.get("star_rng")[:n], dtype=np.float32)
        sfr = np.asarray(hc.get("sfr_rng")[:n], dtype=np.float32)
        xr = np.asarray(hc.get("xray_rng")[:n], dtype=np.float32)
        for lag in (1, 2, 3, 4):
            assert not np.any(sr[:-lag] == sr[lag:]), f"{label} sr==sr lag={lag}"
            assert not np.any(sr[:-lag] == sfr[lag:]), f"{label} sr==sfr lag={lag}"
            assert not np.any(sfr[:-lag] == sr[lag:]), f"{label} sfr==sr lag={lag}"
            assert not np.any(sr[:-lag] == xr[lag:]), f"{label} sr==xr lag={lag}"
            assert not np.any(xr[:-lag] == sr[lag:]), f"{label} xr==sr lag={lag}"
            assert not np.any(sfr[:-lag] == xr[lag:]), f"{label} sfr==xr lag={lag}"
            assert not np.any(xr[:-lag] == sfr[lag:]), f"{label} xr==sfr lag={lag}"
        if label == "desc":
            for prop_name, arr in (
                ("star_rng", sr),
                ("sfr_rng", sfr),
                ("xray_rng", xr),
            ):
                ac = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
                assert abs(ac) < 0.05, f"{label} {prop_name} lag-1 corr={ac}"
                assert abs(float(arr.mean())) < 0.05, f"{label} {prop_name} mean"
                assert 0.95 < float(arr.std()) < 1.05, f"{label} {prop_name} std"
