=====================
CUDA GPU Acceleration
=====================

``21cmFAST`` includes an optional native **C/CUDA + cuFFT** GPU backend built directly
into the compiled CFFI extension. It requires **no CuPy and no runtime NVRTC JIT
compilation**: CPU-only builds remain the default when ``USE_CUDA`` is unset or ``0``,
linking zero CUDA headers or libraries.

Requirements
------------

* **Operating system**: Linux (macOS is not supported for ``USE_CUDA=1``; building with
  ``USE_CUDA=1`` on ``darwin`` raises a ``RuntimeError``).
* **NVIDIA GPU**: Volta or newer (Compute Capability >= 7.0 / ``sm_70``–``sm_90``;
  CUDA >= 13.0 defaults start at Turing ``sm_75``).
* **CUDA Toolkit**: NVIDIA CUDA Toolkit >= 11.2 (provides ``nvcc``, ``libcudart``, and
  ``libcufft``).
* **Host compiler**: ``gcc``/``g++`` or ``clang``/``clang++`` compatible with your ``nvcc``
  version (override with ``CUDAHOSTCXX`` if needed).

Building with CUDA
------------------

Enable the GPU backend at compile time by setting ``USE_CUDA=1`` when building the C
extension (for example, in an editable install or wheel build):

.. code-block:: bash

    USE_CUDA=1 pip install --no-build-isolation -e .

To rebuild only the CFFI extension in place inside a checkout:

.. code-block:: bash

    USE_CUDA=1 python build_cffi.py

Toolchain Resolution & Architecture Flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``build_cffi.py`` resolves the CUDA toolchain and architecture targets automatically,
with explicit overrides for HPC/module and Conda environments:

* ``USE_CUDA``: truthy values (``1``, ``true``, ``yes``, ``on``) compile ``.cu`` sources
  under ``src/py21cmfast/src/cuda/`` with ``nvcc``, perform device linking
  (``nvcc -dlink``), and link ``cudart``, ``cufft``, and ``stdc++`` with
  ``-Wl,-rpath,<cuda_lib>``. Unset or ``0`` compiles CPU-only stubs.
* ``NVCC`` / ``CUDA_HOME`` / ``CUDA_PATH``: ``build_cffi.py`` checks ``NVCC`` first, then
  ``CUDA_HOME`` or ``CUDA_PATH`` (using ``<root>/bin/nvcc``), then ``nvcc`` on ``PATH``,
  and finally ``/usr/local/cuda/bin/nvcc``. Headers are taken from ``<cuda_root>/include``
  and libraries from ``<cuda_root>/lib64`` (falling back to ``<cuda_root>/lib``).
* ``CUDA_ARCHS``: whitespace-, comma-, or semicolon-separated SM architectures (for
  example ``CUDA_ARCHS="80 86"`` or ``sm_80,compute_89``). When unset, ``build_cffi.py``
  selects defaults from ``nvcc --version``:

  * CUDA >= 13.0: ``75 80 86 89 90``
  * CUDA >= 12.0: ``70 75 80 86 89 90``
  * CUDA >= 11.8: ``70 75 80 86 89``
  * Earlier CUDA toolkits: ``70 75 80 86``

  Forward-compatible PTX (``compute_<last>``) is always emitted for the highest
  selected architecture.
* ``CUDAHOSTCXX``: optional host C++ compiler path forwarded to ``nvcc -ccbin``.
* ``*_INC`` and ``*_LIB``: any environment variable whose name contains ``inc`` or
  ``lib`` (case-insensitive, e.g., ``GSL_INC``, ``FFTW_INC``, ``GSL_LIB``,
  ``FFTW_LIB``) supports ``os.pathsep`` (``:``)-separated lists of directories and is
  forwarded to both the host compiler and ``nvcc -I...``.
* ``DEBUG`` and ``LOG_LEVEL``: setting ``DEBUG`` compiles CUDA kernels with
  ``-g -G -O0`` instead of ``-O3``; ``LOG_LEVEL`` sets ``-DLOG_LEVEL=...``.

Common Build Environments
~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

    .. tab-item:: Standard Linux

        .. code-block:: bash

            USE_CUDA=1 pip install --no-build-isolation -e .

    .. tab-item:: Conda Toolchain

        When system headers or libraries live alongside a Conda environment (for example
        ``gcc``/``g++``, ``gsl``, ``fftw`` in ``$CONDA_PREFIX`` plus system CUDA):

        .. code-block:: bash

            export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
            export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
            export GSL_INC="$CONDA_PREFIX/include"
            export GSL_LIB="$CONDA_PREFIX/lib"
            export FFTW_INC="$CONDA_PREFIX/include"
            export FFTW_LIB="$CONDA_PREFIX/lib"
            export CUDA_HOME=/usr/local/cuda
            export USE_CUDA=1
            export CUDA_ARCHS="80 86 89"
            pip install --no-build-isolation -e .

    .. tab-item:: HPC Modules

        On clusters using environment modules, point ``CUDA_HOME`` (or ``NVCC``) to the
        loaded CUDA module, restrict ``CUDA_ARCHS`` to the cluster GPU architecture to
        minimize compile time, and ensure ``LD_LIBRARY_PATH`` includes your runtime
        CUDA/FFTW/GSL shared libraries if they are outside standard system paths:

        .. code-block:: bash

            module load cuda/12.6
            export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
            export CUDA_ARCHS="80"
            export USE_CUDA=1
            pip install --no-build-isolation -e .

Runtime Configuration
---------------------

Compiled CUDA support is **off by default at runtime** so existing CPU workflows behave
identically unless opted in. You can toggle the GPU backend from four places:

1. **Environment variables** (read into ``py21cmfast.config`` on import; truthy values:
   ``1``, ``true``, ``yes``, ``on``):

   .. code-block:: bash

       export P21_USE_CUDA=1
       export P21_CUDA_FAST_HALO_SAMPLER=1   # optional (see below)

2. **Python API** (persistent or temporarily scoped with ``config.use(...)``):

   .. code-block:: python

       import py21cmfast as p21

       # Global toggle in the current process
       p21.config["USE_CUDA"] = True

       # Temporarily scoped override (automatically restored on exit)
       with p21.config.use(USE_CUDA=True, CUDA_FAST_HALO_SAMPLER=False):
           lc = p21.run_lightcone(...)

3. **YAML configuration file** (via ``config.write`` and ``Config.load``):

   .. code-block:: yaml

       USE_CUDA: true
       CUDA_FAST_HALO_SAMPLER: false

   .. code-block:: python

       import py21cmfast as p21
       from py21cmfast._cfg import Config

       p21.config.write("config.yml")
       p21.config.update(Config.load("config.yml"))

4. **Command-line interface (CLI)**:

   The ``21cmfast run`` subcommands (``ics``, ``coeval``, ``lightcone``, ``global``) and
   ``21cmfast dev feature`` accept ``--use-cuda`` / ``--no-use-cuda`` and
   ``--cuda-fast-halo-sampler`` / ``--no-cuda-fast-halo-sampler``:

   .. code-block:: bash

       21cmfast run lightcone --template latest --use-cuda --out lightcone.h5
       21cmfast run coeval -z 8 10 --template latest --use-cuda --cuda-fast-halo-sampler
       21cmfast run ics --template simple tiny --no-use-cuda

   When omitted, CLI commands preserve the active ``py21cmfast.config`` values; when
   supplied, they apply cleanly for the simulation call and restore prior values on exit.

Supported Stages & Automatic CPU Fallback
-----------------------------------------

The native CUDA backend accelerates all core C simulation stages while keeping arrays
device-resident via a lazy host-pointer registry:

* ``InitialConditions`` (FFT power-spectrum realization, velocity/2LPT derivatives,
  resampling)
* ``PerturbedField`` (Zel'dovich / 2LPT particle displacement + mass deposition +
  smoothing)
* ``HaloCatalog`` (``DEXM-ESF`` excursion-set finder and ``CHMF-SAMPLER`` mass sampling)
  and ``PerturbedHaloCatalog``
* ``EmissivityFields`` (Eulerian and Lagrangian emissivity grids)
* ``RadiationFields`` (batched multi-shell cuFFT convolutions and optical-depth
  filtering)
* ``TsBox``, ``IonizedBox`` (including inhomogeneous recombinations), and
  ``BrightnessTemp``
* Conditional ``dNdM`` interpolation tables

**Automatic CPU OpenMP fallback** occurs transparently whenever:

* ``21cmFAST`` was compiled with ``USE_CUDA=0`` (or unset),
* ``py21cmfast.config["USE_CUDA"]`` is ``False``,
* No CUDA GPU is visible at runtime (e.g., ``CUDA_VISIBLE_DEVICES=""`` or CPU login node),
* A specific simulation mode falls outside GPU kernel assumptions (such as special CPU-only
  diagnostic paths).

Fast GPU Halo Sampler (``CUDA_FAST_HALO_SAMPLER``)
--------------------------------------------------

When ``SOURCE_MODEL="CHMF-SAMPLER"``, enabling ``USE_CUDA=True`` with
``CUDA_FAST_HALO_SAMPLER=False`` (the default) keeps ``HaloCatalog`` sampling on the
CPU GSL sampler (preserving CPU-compatible RNG ordering while the GPU accelerates
``dNdM`` tables, ``PerturbedHaloCatalog``, ``EmissivityFields``, ``RadiationFields``,
``TsBox``, ``IonizedBox``, and ``BrightnessTemp``). Enabling
``CUDA_FAST_HALO_SAMPLER=True`` (or ``P21_CUDA_FAST_HALO_SAMPLER=1`` /
``--cuda-fast-halo-sampler``) dispatches grid and progenitor sampling to a parallel GPU
**Philox-4x32-10** PRNG (``p21_cuda_sample_halo_grids`` /
``p21_cuda_sample_halo_progenitors``), eliminating host-side RNG bottlenecks at high
resolution while remaining statistically consistent with the default sampler.

Multiprocessing & ``21CMMC`` Workers
------------------------------------

CUDA context initialization inside ``21cmFAST`` is **lazy**: importing ``py21cmfast`` does
not create a CUDA context on import. For multi-worker parameter sweeps (such as
``21CMMC`` or ``multiprocessing``):

* Prefer initializing CUDA only inside worker processes after ``fork()``, or use the
  ``spawn`` start method.
* Assign one GPU per worker where possible via ``CUDA_VISIBLE_DEVICES`` (e.g., mapping
  worker rank modulo available GPUs).
* Call ``lib.p21_cuda_cleanup()`` between independent runs if you need to flush cached
  device arrays, destroy cuFFT plans, trim the device memory pool, and release the
  active CUDA stream.

Diagnostics & Stats API
-----------------------

You can inspect compile-time status, runtime availability, memory usage, and per-stage
GPU execution counters through the compiled CFFI library:

.. code-block:: python

    from py21cmfast.c_21cmfast import ffi, lib

    print("Compiled with CUDA:", bool(lib.p21_cuda_is_compiled()))
    print("GPU visible now:   ", bool(lib.p21_cuda_is_available()))
    print("CUDA enabled:      ", bool(lib.p21_cuda_get_enabled()))
    print("Fast halo sampler: ", bool(lib.p21_cuda_get_fast_halo_sampler()))

    # Reset and read P21CudaStats after a run
    lib.p21_cuda_reset_stats()
    # ... run simulation ...
    stats = ffi.new("P21CudaStats *")
    lib.p21_cuda_get_stats(stats)
    print(
        f"device={stats.device_id} SMs={stats.sm_count} "
        f"free={stats.free_mem_bytes / 1e9:.2f} GB "
        f"ic_runs={stats.ic_runs} perturb_runs={stats.perturb_runs} "
        f"ts_runs={stats.spin_temp_runs} ion_runs={stats.ionization_runs}"
    )

Key ``P21CudaStats`` fields include device identity (``compiled``, ``enabled``,
``device_id``, ``sm_count``, ``free_mem_bytes``, ``total_mem_bytes``), memory/cache
activity (``mempool_alloc_calls``, ``registry_hits``, ``registry_misses``,
``cufft_plan_hits``, ``cufft_plan_misses``), and stage counters (``ic_runs``,
``perturb_runs``, ``emissivity_runs``, ``radiation_setup_runs``,
``radiation_update_runs``, ``spin_temp_runs``, ``ionization_runs``,
``brightness_runs``, ``halo_dexm_runs``, ``halo_sampler_runs``,
``halo_perturb_runs``, ``dndm_table_runs``).

Performance Summary
-------------------

On an **NVIDIA RTX A6000** (``sm_86``, CUDA 12.6), measured end-to-end simulation wall
times across major ``21cmFAST`` source models are:

* **Eulerian lightcone** (``192^3 / 64^3``, 43 redshift nodes, ``BOX_LEN=200 Mpc``,
  ``USE_TS_FLUCT=True``, ``RECOMB_MODEL="inhomogeneous"``):
  ``7.849 s`` (CUDA warm 2T) vs. ``111.371 s`` (CPU 1T, **12.09x** vs. CUDA 1T ``9.210 s``),
  ``61.504 s`` (CPU 2T, **7.84x**), and ``37.801 s`` (CPU 4T, **4.82x**).
* **Lagrangian ``L-INTEGRAL``** (``192^3 / 64^3``, 10 nodes, 2T):
  ``2.687 s`` (CUDA warm) vs. ``11.786 s`` (CPU, **4.39x**).
* **``DEXM-ESF``** (``96^3 / 32^3``, 3 nodes, 2T):
  ``0.804 s`` (CUDA warm) vs. ``3.198 s`` (CPU, **3.98x**).
* **``CHMF-SAMPLER``** (``96^3 / 32^3``, 5 nodes, 2T):
  ``2.429 s`` (CUDA default, **4.32x** vs. CPU 2T ``10.499 s``) and ``1.643 s``
  (``CUDA_FAST_HALO_SAMPLER=True``, **6.39x** vs. CPU 2T).
