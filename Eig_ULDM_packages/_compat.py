"""Small compatibility helpers shared by the toolkit modules.

Everything here is dependency-light (NumPy plus the standard library; SciPy
only inside :func:`sph_harm_ylm`) so that it can be imported from any module
without changing import-time behaviour.

Contents
--------
- :func:`trapezoid`: ``np.trapezoid`` (NumPy >= 2.0) or ``np.trapz`` (NumPy < 2.0).
  Both implement the same composite trapezoidal formula, so results are
  identical to the historical ``np.trapz`` call on the same NumPy version.
- :func:`sph_harm_ylm`: orthonormal complex Y_lm with the Condon-Shortley phase
  from SciPy, using ``sph_harm_y(l, m, theta, phi)`` (SciPy >= 1.15) or the
  removed ``sph_harm(m, l, phi, theta)`` (argument order differs!).
- :func:`mp_context`: a multiprocessing context that works on Linux, macOS and
  Windows ('fork' only where it is the safe platform default, else 'spawn').
- :func:`default_workers`: worker count from the environment / CPU affinity
  instead of a hard-coded core count.
- :func:`prefix_folder`: first path component of a file prefix, portable
  across ``/`` and ``\\`` separators.
"""

from __future__ import annotations

import os
import sys
import multiprocessing as _mp
from typing import Optional

import numpy as np

__all__ = [
    "trapezoid",
    "sph_harm_ylm",
    "mp_context",
    "default_workers",
    "prefix_folder",
]

# ---------------------------------------------------------------------------
# NumPy 2 removed np.trapz; np.trapezoid does not exist before NumPy 2.0.
# ---------------------------------------------------------------------------
if hasattr(np, "trapezoid"):
    trapezoid = np.trapezoid
else:  # pragma: no cover - exercised on NumPy < 2.0 (py39-cluster)
    trapezoid = np.trapz


def sph_harm_ylm(l: int, m: int, theta, phi):
    """Orthonormal complex spherical harmonic Y_l^m(theta, phi) from SciPy.

    ``theta`` is the polar (colatitude) angle in [0, pi] and ``phi`` the
    azimuth in [0, 2 pi), both in radians.  The Condon-Shortley phase is
    included, matching ``pyshtools.expand.spharm_lm(..., normalization='ortho',
    kind='complex', csphase=-1)``.

    SciPy >= 1.15 provides ``sph_harm_y(n, m, theta, phi)``; older SciPy only
    has ``sph_harm(m, n, phi, theta)`` (removed in SciPy 1.17) whose argument
    order is different.  This wrapper hides that difference.
    """
    import scipy.special as _ss

    if hasattr(_ss, "sph_harm_y"):
        return _ss.sph_harm_y(int(l), int(m), theta, phi)
    return _ss.sph_harm(int(m), int(l), phi, theta)  # pragma: no cover - SciPy < 1.15


def mp_context(method: Optional[str] = None):
    """Return a multiprocessing context that is safe on the current platform.

    Priority: explicit ``method`` > environment variable
    ``ULDM_MP_START_METHOD`` > platform default.  The platform default is
    'fork' on Linux (the historical behaviour of this toolkit, which used
    ``get_context('fork')`` unconditionally) and 'spawn' everywhere else
    ('fork' does not exist on Windows and is unsafe on macOS).  Worker
    functions and their initargs must therefore be picklable, and scripts
    that start pools must use an ``if __name__ == "__main__":`` guard.
    """
    if method is None:
        method = os.environ.get("ULDM_MP_START_METHOD") or None
    available = _mp.get_all_start_methods()
    if method is None:
        method = "fork" if (sys.platform.startswith("linux") and "fork" in available) else "spawn"
    if method not in available:
        raise ValueError(
            f"multiprocessing start method {method!r} is not available on this platform "
            f"(available: {available})."
        )
    return _mp.get_context(method)


def _int_env(name: str) -> Optional[int]:
    val = os.environ.get(name, "").strip()
    if not val:
        return None
    try:
        n = int(val.split(",")[0].split("(")[0])
    except ValueError:
        return None
    return n if n > 0 else None


def available_cpus() -> int:
    """CPUs this process may use (CPU affinity when available, else cpu_count)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def default_workers(explicit: Optional[int] = None, env_names=None) -> int:
    """Number of workers to use.

    Priority: ``explicit`` (if a positive int) > the first set environment
    variable among ``ULDM_NUM_WORKERS``, ``PBS_NCPUS``, ``NCPUS``,
    ``SLURM_CPUS_PER_TASK``, ``OMP_NUM_THREADS`` > CPU affinity of this
    process > ``os.cpu_count()``.  Scheduler variables are only used as an
    upper bound when they exceed the CPUs actually available.
    """
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    if env_names is None:
        env_names = ("ULDM_NUM_WORKERS", "PBS_NCPUS", "NCPUS", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS")
    avail = available_cpus()
    for name in env_names:
        n = _int_env(name)
        if n is not None:
            return max(1, min(n, avail)) if name != "ULDM_NUM_WORKERS" else n
    return avail


def prefix_folder(file_prefix: str) -> str:
    """First path component of a snapshot file prefix.

    ``"3WfnRS/P3R_#"`` -> ``"3WfnRS"`` (identical to the historical
    ``file_prefix.split('/')[0]`` for POSIX relative prefixes), but also
    ``"3WfnRS\\P3R_#"`` -> ``"3WfnRS"`` on Windows.  For an absolute prefix the
    name of the directory that contains the snapshots is returned (the
    historical code returned ``''`` or a drive letter there, which put the
    JSON files in the wrong place).
    """
    p = str(file_prefix).replace("\\", "/")
    if os.path.isabs(str(file_prefix)) or p.startswith("/"):
        parent = os.path.dirname(os.path.normpath(str(file_prefix)))
        return os.path.basename(parent) or parent
    first = p.split("/")[0]
    return first


def legacy_prefix_folder(file_prefix: str) -> str:
    """The folder name toolkit 0.2.0 used: ``file_prefix.split('/')[0]`` (``''`` for a POSIX absolute prefix)."""
    return str(file_prefix).split("/")[0]


def alm_json_path(file_prefix: str, r_0, l_max: int, filename: str) -> str:
    """Path of one per-shell a_lm JSON file below ``alm_coefficients/`` (relative to the cwd).

    New files go to ``alm_coefficients/<prefix_folder(file_prefix)>/r0_<r_0>/lmax_<l_max>/<filename>``.
    For an absolute prefix, toolkit 0.2.0 wrote them under ``alm_coefficients/<file_prefix.split('/')[0]>/...``
    (``alm_coefficients/r0_<r_0>/...`` on POSIX).  When the new file does not exist but such a historical
    file does, the historical path is returned, so 0.2.0 trees stay readable.
    """
    tail = (f"r0_{r_0}", f"lmax_{l_max}", filename)
    new = os.path.join("alm_coefficients", prefix_folder(file_prefix), *tail)
    if os.path.exists(new):
        return new
    legacy_folder = legacy_prefix_folder(file_prefix)
    if legacy_folder != prefix_folder(file_prefix):
        old = os.path.join("alm_coefficients", legacy_folder, *tail)
        if os.path.exists(old):
            return old
    return new
