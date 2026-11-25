"""Build the C code with CFFI."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from cffi import FFI


# Get the compiler. We support gcc and clang.
# The compiler is determnined from the environment and uses sysconfig as a fallback.
def detect_default_compiler():
    """Detect the default compiler using its "--version" command.

    CFFI gives us no control over which compiler is used, so we have to
    detect which compiler is being used based on the environment or sysconfig.

    Additionally, it uses the deprecated distutils package to get the default compiler,
    so we avoid that here. Instead calling the compiler with "--version" and looking for
    substrings.

    Returns a string indicating the compiler type: "gcc", "clang", or "unknown".
    This allows us to set compiler-specific flags later on.
    """
    import sysconfig
    import warnings
    from subprocess import run

    source = "environment variable 'CC'" if "CC" in os.environ else "sysconfig"
    _compiler = os.environ.get("CC", sysconfig.get_config_var("CC"))
    if not _compiler:
        warnings.warn(
            "No compiler was found via CC or sysconfig('CC'). "
            "No special compiler flags will be set.",
            stacklevel=2,
        )
        return "unknown"

    # strip any arguments from the compiler command
    _compiler = _compiler.split()[0]

    print(f"Using compiler from {source}: {_compiler}")

    result = run([_compiler, "--version"], capture_output=True, text=True, check=False)
    compiler_banner = f"{result.stdout}\n{result.stderr}".lower()

    # Check for clang first: clang banners often include "linux-gnu", which
    # would otherwise be misclassified as gcc by a naive "gnu" substring check.
    if "clang" in _compiler.lower() or "clang" in compiler_banner:
        return "clang"
    if _compiler.lower() == "gcc" or any(
        s in compiler_banner for s in ["gcc", "g++", "gnu compiler collection"]
    ):
        return "gcc"
    else:
        warnings.warn(
            f"21cmFAST was not able to determine the compiler type from '{_compiler}'."
            " No special compiler flags will be set.",
            stacklevel=2,
        )
    return "unknown"


ffi = FFI()

LOCATION = Path(__file__).resolve().parent
CLOC = LOCATION / "src" / "py21cmfast" / "src"
include_dirs = [str(CLOC)]
c_files = [str(fl.relative_to(LOCATION)) for fl in sorted(CLOC.glob("*.c"))]

# Set the C-code logging level.
# If DEBUG is set, we default to the highest level, but if not,
# we set it to the level just above no logging at all.
log_level = os.environ.get("LOG_LEVEL", "4" if "DEBUG" in os.environ else "1")
available_levels = [
    "NONE",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "SUPER_DEBUG",
    "ULTRA_DEBUG",
]


if isinstance(log_level, str) and log_level.upper() in available_levels:
    log_level = available_levels.index(log_level.upper())

try:
    log_level = int(log_level)
except ValueError as e:
    # note: for py35 support, can't use f strings.
    raise ValueError(
        "LOG_LEVEL must be specified as a positive integer, or one "
        f"of {available_levels}"
    ) from e

# ==================================================
# Set compilation arguments dependent on environment
# ==================================================

extra_compile_args = ["-Wall", "--verbose", f"-DLOG_LEVEL={log_level:d}"]

if "DEBUG" in os.environ:
    extra_compile_args += ["-g", "-O0"]
else:
    extra_compile_args += ["-Ofast"]

libraries = ["m", "gsl", "gslcblas", "fftw3f_omp", "fftw3f"]

# stuff for gperftools
if "PROFILE" in os.environ:
    libraries += ["profiler", "tcmalloc"]
    # we need this even if DEBUG is off
    extra_compile_args += ["-g"]

# do compiler-specific stuff here
compiler = detect_default_compiler()
if compiler == "clang":
    libraries += ["omp"]
elif compiler == "gcc":
    libraries += ["gomp"]

if sys.platform == "darwin":
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
elif sys.platform.startswith("linux"):
    if compiler == "clang":
        extra_compile_args += ["-fopenmp=libomp"]
    else:
        extra_compile_args += ["-fopenmp"]

library_dirs = []
for k, v in os.environ.items():
    if not v:
        continue
    if "inc" in k.lower():
        include_dirs += [p for p in v.split(os.pathsep) if p]
    elif "lib" in k.lower():
        library_dirs += [p for p in v.split(os.pathsep) if p]


def _resolve_cuda_toolchain() -> tuple[Path, Path, Path]:
    """Locate ``nvcc`` and the corresponding CUDA include and library directories.

    Returns
    -------
    nvcc : Path
        Path to the ``nvcc`` executable.
    cuda_inc : Path
        Path to the CUDA header directory.
    cuda_lib : Path
        Path to the CUDA library directory.
    """
    nvcc_env = os.environ.get("NVCC")
    cuda_home_env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if nvcc_env:
        nvcc = Path(nvcc_env)
    elif cuda_home_env:
        nvcc = Path(cuda_home_env) / "bin" / "nvcc"
    else:
        which_nvcc = shutil.which("nvcc")
        nvcc = Path(which_nvcc) if which_nvcc else Path("/usr/local/cuda/bin/nvcc")

    if not nvcc.exists():
        raise FileNotFoundError(
            f"USE_CUDA=1 requested, but nvcc was not found at '{nvcc}'. "
            "Set NVCC or CUDA_HOME to a valid CUDA toolkit."
        )

    cuda_root = (
        Path(cuda_home_env).resolve() if cuda_home_env else nvcc.resolve().parent.parent
    )
    cuda_inc = cuda_root / "include"
    cuda_lib = cuda_root / "lib64"
    if not cuda_lib.exists():
        cuda_lib = cuda_root / "lib"
    return nvcc, cuda_inc, cuda_lib


def _detect_nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    """Detect the major and minor CUDA toolkit version from ``nvcc --version``."""
    result = subprocess.run(
        [str(nvcc), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    banner = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"release\s+(\d+)\.(\d+)", banner)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_cuda_archs(nvcc_version: tuple[int, int] | None = None) -> list[str]:
    """Parse ``CUDA_ARCHS`` into numeric SM architecture strings."""
    raw = os.environ.get("CUDA_ARCHS", "").strip()
    if raw:
        tokens = raw.replace(",", " ").replace(";", " ").split()
        archs: list[str] = []
        for tok in tokens:
            cleaned = tok.removeprefix("sm_").removeprefix("compute_")
            if cleaned and cleaned not in archs:
                archs.append(cleaned)
        if archs:
            return archs

    if nvcc_version is None:
        return ["75", "80", "86", "89"]
    if nvcc_version >= (13, 0):
        return ["75", "80", "86", "89", "90"]
    if nvcc_version >= (12, 0):
        return ["70", "75", "80", "86", "89", "90"]
    if nvcc_version >= (11, 8):
        return ["70", "75", "80", "86", "89"]
    return ["70", "75", "80", "86"]


def _compile_cuda_objects(
    cloc: Path,
    obj_dir: Path,
    log_level: int,
    include_dirs: list[str],
    library_dirs: list[str],
    libraries: list[str],
    extra_compile_args: list[str],
) -> tuple[list[str], list[str]]:
    """Compile native CUDA sources and perform device linking when ``USE_CUDA=1``.

    Parameters
    ----------
    cloc : Path
        Root directory of C/CUDA sources.
    obj_dir : Path
        Directory where compiled ``.o`` artifacts are cached.
    log_level : int
        Preprocessor ``LOG_LEVEL`` integer passed to ``nvcc``.
    include_dirs : list[str]
        Include directory list updated with CUDA headers.
    library_dirs : list[str]
        Library directory list updated with CUDA libraries.
    libraries : list[str]
        Linked library list updated with CUDA runtime libraries.
    extra_compile_args : list[str]
        Host C compiler flags updated with ``-DUSE_CUDA=1``.

    Returns
    -------
    extra_objects : list[str]
        Compiled CUDA object files including the ``nvcc -dlink`` object.
    extra_link_args : list[str]
        Linker flags such as ``-Wl,-rpath`` for the CUDA library directory.
    """
    if sys.platform == "darwin":
        raise RuntimeError(
            "USE_CUDA=1 is not supported on macOS; build with USE_CUDA=0."
        )

    nvcc, cuda_inc, cuda_lib = _resolve_cuda_toolchain()
    nvcc_version = _detect_nvcc_version(nvcc)
    archs = _parse_cuda_archs(nvcc_version)
    arch_flags: list[str] = []
    for arch in archs:
        arch_flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    arch_flags.extend(
        ["-gencode", f"arch=compute_{archs[-1]},code=compute_{archs[-1]}"]
    )

    if str(cuda_inc) not in include_dirs:
        include_dirs.append(str(cuda_inc))
    if str(cuda_lib) not in library_dirs:
        library_dirs.append(str(cuda_lib))

    extra_compile_args.append("-DUSE_CUDA=1")
    for lib_name in ("cudart", "cufft", "stdc++"):
        if lib_name not in libraries:
            libraries.append(lib_name)

    cu_dir = cloc / "cuda"
    cu_files = sorted(cu_dir.glob("*.cu"))
    if not cu_files:
        raise FileNotFoundError(
            f"USE_CUDA=1 requested, but no .cu sources found in '{cu_dir}'."
        )

    debug_flags = ["-g", "-G", "-O0"] if "DEBUG" in os.environ else ["-O3"]
    cuda_host_cxx = os.environ.get("CUDAHOSTCXX", "").strip()
    ccbin_flags = ["-ccbin", cuda_host_cxx] if cuda_host_cxx else []
    nvcc_includes = list(dict.fromkeys([str(cloc), str(cu_dir), *include_dirs]))
    nvcc_flags = [
        *ccbin_flags,
        "-Xcompiler",
        "-fPIC",
        "-std=c++17",
        "-rdc=true",
        "--fmad=false",
        f"-DLOG_LEVEL={log_level:d}",
        "-DUSE_CUDA=1",
        *debug_flags,
        *arch_flags,
        *(f"-I{inc}" for inc in nvcc_includes),
    ]

    obj_dir.mkdir(parents=True, exist_ok=True)
    stamp_file = obj_dir / ".flags.stamp"
    stamp_text = "\n".join([str(nvcc), cuda_host_cxx, *nvcc_flags])
    flags_changed = not stamp_file.exists() or stamp_file.read_text() != stamp_text

    header_files = [
        *cu_dir.glob("*.cuh"),
        *cu_dir.glob("*.h"),
        *cloc.glob("*.h"),
    ]
    max_header_mtime = max((h.stat().st_mtime for h in header_files), default=0.0)

    obj_paths: list[str] = []
    rebuilt_any = flags_changed
    for cu_file in cu_files:
        obj_file = obj_dir / f"{cu_file.stem}.o"
        obj_paths.append(str(obj_file))
        src_mtime = max(cu_file.stat().st_mtime, max_header_mtime)
        needs_build = (
            flags_changed
            or not obj_file.exists()
            or obj_file.stat().st_mtime < src_mtime
        )
        if needs_build:
            cmd = [
                str(nvcc),
                "-c",
                *nvcc_flags,
                str(cu_file),
                "-o",
                str(obj_file),
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            rebuilt_any = True

    dlink_obj = obj_dir / "cuda_dlink.o"
    if rebuilt_any or not dlink_obj.exists():
        dlink_cmd = [
            str(nvcc),
            "-dlink",
            *ccbin_flags,
            "-Xcompiler",
            "-fPIC",
            *arch_flags,
            *obj_paths,
            "-o",
            str(dlink_obj),
        ]
        print(" ".join(dlink_cmd))
        subprocess.run(dlink_cmd, check=True)
        stamp_file.write_text(stamp_text)

    extra_objects = [*obj_paths, str(dlink_obj)]
    extra_link_args = [f"-Wl,-rpath,{cuda_lib}"]
    return extra_objects, extra_link_args


use_cuda = os.environ.get("USE_CUDA", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
extra_objects: list[str] = []
extra_link_args: list[str] = []
if use_cuda:
    extra_objects, extra_link_args = _compile_cuda_objects(
        cloc=CLOC,
        obj_dir=LOCATION / "build" / "cuda_objs",
        log_level=log_level,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    )

# =================================================================
# NOTES FOR DEVELOPERS:
#   The CFFI implementation works as follows:
#       - All function prototypes, global variables and type definitions *directly* used
#           in the python wrapper must be declared via ffi.cdef("""C CODE""").
#           There must be no compiler directives in this code (#include, #define, etc)
#       - All implementations of global variables and types present in the cdef() calls
#           must also be present in the second argument of set_source.
#           This is passed to the compiler.
#       - The `sources` kwarg then contains all the .c files in the library which are to be compiled

# This is the overall C code.
ffi.set_source(
    "py21cmfast.c_21cmfast",  # Name/Location of shared library module
    """
    #include "21cmFAST.h"
    """,
    sources=c_files,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_objects=extra_objects,
    extra_link_args=extra_link_args,
)

# Header files containing types, globals and function prototypes
with (CLOC / "_inputparams_wrapper.h").open() as f:
    ffi.cdef(f.read())
with (CLOC / "_outputstructs_wrapper.h").open() as f:
    ffi.cdef(f.read())
with (CLOC / "_functionprototypes_wrapper.h").open() as f:
    ffi.cdef(f.read())

# CFFI needs to be able to access a free function to make the __del__ method for OutputStruct fields
#  This will expose the standard free() function to the wrapper so it can be used
ffi.cdef(
    """
        void free(void *ptr);
    """
)

if __name__ == "__main__":
    ffi.compile()
