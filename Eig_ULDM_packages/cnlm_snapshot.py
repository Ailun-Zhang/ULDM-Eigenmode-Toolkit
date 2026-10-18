r"""Per-snapshot c_nlm building blocks and writers for the historical file layouts.

These functions let a caller that produces one snapshot at a time (the local
pipeline :mod:`Eig_ULDM_packages.pipeline`, or a recorder running inside a
PyUL simulation) compute exactly what the historical offline chain computes:

    process_file (JSON) -> create_sh_hdf5 -> load_a_lm -> compute_cnlm -> save_cnlm_to_h5

Bitwise contract
----------------
For one snapshot with flattened coefficients ``alm`` (shape ``(n_r, (l_max+1)**2)``
as returned by :func:`Eig_ULDM_packages.alm_kernel.compute_alm_cube`):

1. :func:`alm_counts_roundtrip` reproduces the storage round trip of the
   a_lm HDF5 file (``ylm = a_lm*counts`` written by ``create_sh_hdf5``, then
   ``ylm/counts`` in ``load_a_lm``).  This round trip is **not** an identity
   in floating point (about 14 % of values change in the last bit), so it must
   be applied to obtain the historical c_nlm bit for bit.
2. :class:`CnlmProjector` then performs, for that snapshot, the same radial
   interpolation (``interp1d`` cubic, constant extrapolation) and the same
   trapezoidal integral on the eigenfunction grid as ``compute_cnlm`` with
   ``grid_choice='eigen'``.
3. :func:`write_cnlm_h5` writes ``c_nlm.h5`` with ``save_cnlm_to_h5`` (root
   ``t`` float64 + ``n/<n>/l/<l>/m_<m>`` complex128), and
   :func:`write_alm_h5` writes the a_lm file in the ``create_sh_hdf5``
   layout.  Provenance goes into root attributes only.

Radial grid
-----------
``compute_c_nlm.ipynb`` builds the code-unit radii of the a_lm shells as
``linspace(0, convert(r_max), n_r)`` with ``r[0] = convert(1e-5 kpc)`` and then
rescales them onto the eigen grid (``r/r[-1]*r_max_eig``).
:func:`legacy_alm_radial_grid` reproduces this, with a consistency check of
the two outer radii instead of a silent rescale.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.interpolate as _interp

from ._compat import trapezoid as _trapezoid
from .alm_kernel import lm_list

__all__ = [
    "alm_counts",
    "alm_counts_roundtrip",
    "legacy_alm_radial_grid",
    "CnlmProjector",
    "cnlm_from_alm_snapshot",
    "stack_cnlm_snapshots",
    "write_alm_h5",
    "write_cnlm_h5",
    "INLINE_SOURCE_ID",
    "kernel_fingerprint",
    "KERNEL_FORMAT_VERSION",
    "checkpoint_path",
    "write_alm_checkpoint",
    "read_alm_checkpoint",
]

#: ``source_id`` of a checkpoint written by the in-simulation recorder
#: (:mod:`Eig_ULDM_packages.cnlm_inline`): the cube came from PyUL's memory, not from a file.
INLINE_SOURCE_ID = "pyul-inline"


def alm_counts(num_points: int) -> int:
    """The historical ``counts`` value ``num_points * 2 * num_points``."""
    return int(num_points) * (2 * int(num_points))


def alm_counts_roundtrip(alm: np.ndarray, num_points: int) -> np.ndarray:
    """Apply the a_lm HDF5 storage round trip of the historical chain.

    Performs ``ylm = alm * counts`` (as ``create_sh_hdf5``) and then
    ``where(counts != 0, ylm / counts, 0)`` with a float64 ``counts`` array
    (as ``load_a_lm``), returning the a_lm values that ``compute_cnlm`` sees
    in the historical chain.
    """
    counts = alm_counts(num_points)
    ylm = np.asarray(alm, dtype=np.complex128) * counts
    counts_arr = np.full(ylm.shape, counts, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(counts_arr != 0, ylm / counts_arr, 0.0)


def legacy_alm_radial_grid(
    r_max: float,
    n_r: int,
    r_max_eig: float,
    *,
    r0: float = 1e-5,
    length_units: str = "kpc",
    rtol: float = 1e-6,
) -> np.ndarray:
    """Code-unit radii of the a_lm shells exactly as in ``compute_c_nlm.ipynb``.

    ``r_max`` and ``r0`` are in ``length_units`` (the units of the shells
    used by the a_lm kernel); conversion uses
    :func:`Eig_ULDM_packages.units.convert_between` with the axion mass
    currently set in :mod:`Eig_ULDM_packages.units`.  The final rescale onto
    ``r_max_eig`` is applied only after :func:`check_rmax_consistency`
    confirmed that the two outer radii agree within ``rtol``.
    """
    from . import units
    from .c_nlm_integrator import rescale_r_to_eigen_rmax

    r_max_code_units = units.convert_between(r_max, length_units, '', 'l')
    r_values = np.linspace(0.0, r_max_code_units, int(n_r), endpoint=True)
    r_values[0] = units.convert_between(r0, length_units, '', 'l')
    return rescale_r_to_eigen_rmax(r_values, r_max_eig, rtol=rtol)


class CnlmProjector:
    """Project one snapshot's a_lm(r) onto the radial eigenfunctions.

    Parameters
    ----------
    r_f : 1-D array
        Eigenfunction grid (code units), e.g. from ``load_eigenfunctions``.
    f_dict : dict
        ``{(l, n): f_nl(r_f)}`` (complex128 as returned by
        ``load_eigenfunctions``; real arrays are cast the same way).
    r_al : 1-D array
        Radii of the a_lm shells in the units of ``r_f``.
    l_max : int
        Maximum degree present in the a_lm arrays passed to :meth:`project`.
    r_interp_kind : str
        ``interp1d`` kind for a_lm(r) -> r_f (default 'cubic', as the paper).

    The projection uses ``grid_choice='eigen'`` of ``compute_cnlm``:
    ``c_nlm = trapezoid(r_f**2 * f_nl * interp(a_lm)(r_f), dx=r_f[1]-r_f[0])``.
    Keys are produced in the order of the serial ``compute_cnlm`` loop.
    """

    def __init__(
        self,
        r_f: np.ndarray,
        f_dict: Mapping[Tuple[int, int], np.ndarray],
        r_al: np.ndarray,
        l_max: int,
        r_interp_kind: str = "cubic",
    ) -> None:
        self.r_f = np.asarray(r_f, dtype=np.float64)
        self.r_al = np.asarray(r_al, dtype=np.float64)
        if self.r_f.ndim != 1 or self.r_al.ndim != 1:
            raise ValueError("r_f and r_al must be one-dimensional arrays.")
        if not np.all(np.diff(self.r_f) > 0):
            raise ValueError("r_f must be strictly increasing.")
        if not np.all(np.diff(self.r_al) > 0):
            raise ValueError("r_al must be strictly increasing.")
        self.l_max = int(l_max)
        self.r_interp_kind = r_interp_kind
        self.weight = self.r_f ** 2
        self.dr_int = self.r_f[1] - self.r_f[0]
        self.lm = lm_list(self.l_max)
        self._col = {lm: i for i, lm in enumerate(self.lm)}
        # weight * f exactly as the first product of compute_cnlm's integrand
        self._wf: Dict[Tuple[int, int], np.ndarray] = {}
        self._order: List[Tuple[int, int, int]] = []
        for (ell, n_idx) in sorted(f_dict.keys()):
            f_vec = np.asarray(f_dict[(ell, n_idx)])
            if not np.iscomplexobj(f_vec):
                f_vec = f_vec.astype(np.complex128)
            if f_vec.shape != self.r_f.shape:
                raise ValueError(f"f_dict[{(ell, n_idx)}] has shape {f_vec.shape}, expected {self.r_f.shape}.")
            if ell > self.l_max:
                continue
            self._wf[(ell, n_idx)] = self.weight * f_vec
            for m_val in range(-ell, ell + 1):
                self._order.append((ell, n_idx, m_val))
        self._kind = r_interp_kind
        if r_interp_kind == "cubic" and self.r_al.size < 4:
            self._kind = "linear"
        elif r_interp_kind == "quadratic" and self.r_al.size < 3:
            self._kind = "linear"

    @property
    def keys(self) -> List[Tuple[int, int, int]]:
        """(n, l, m) keys in output order."""
        return [(n, l, m) for (l, n, m) in self._order]

    def project(self, alm: np.ndarray) -> Dict[Tuple[int, int, int], complex]:
        """c_nlm of one snapshot from ``alm`` of shape ``(n_r, (l_max+1)**2)``.

        Pass the output of :func:`alm_counts_roundtrip` to reproduce the
        historical chain bit for bit.
        """
        alm = np.asarray(alm)
        if alm.shape != (self.r_al.size, len(self.lm)):
            raise ValueError(f"alm has shape {alm.shape}, expected {(self.r_al.size, len(self.lm))}.")
        a_interp_cache: Dict[Tuple[int, int], np.ndarray] = {}
        out: Dict[Tuple[int, int, int], complex] = {}
        for (ell, n_idx, m_val) in self._order:
            a_interp = a_interp_cache.get((ell, m_val))
            if a_interp is None:
                a_slice = np.array(alm[:, self._col[(ell, m_val)]], dtype=np.complex128)
                interp_func = _interp.interp1d(
                    self.r_al,
                    a_slice,
                    kind=self._kind,
                    bounds_error=False,
                    fill_value=(a_slice[0], a_slice[-1]),
                    assume_sorted=True,
                )
                a_interp = interp_func(self.r_f)
                a_interp_cache[(ell, m_val)] = a_interp
            integrand = self._wf[(ell, n_idx)] * a_interp
            out[(n_idx, ell, m_val)] = _trapezoid(integrand, dx=self.dr_int)
        return out


def cnlm_from_alm_snapshot(
    alm: np.ndarray,
    r_al: np.ndarray,
    r_f: np.ndarray,
    f_dict: Mapping[Tuple[int, int], np.ndarray],
    *,
    l_max: Optional[int] = None,
    r_interp_kind: str = "cubic",
    num_points_roundtrip: Optional[int] = None,
) -> Dict[Tuple[int, int, int], complex]:
    """c_nlm of ONE snapshot, as ``compute_cnlm`` computes it for that time index.

    Parameters
    ----------
    alm : array (n_r, (l_max+1)**2)
        Flattened a_lm of the snapshot (columns l = 0..l_max, m = -l..l).
    r_al, r_f, f_dict, r_interp_kind :
        See :class:`CnlmProjector`.
    l_max : int, optional
        Inferred from ``alm.shape[1]`` when omitted.
    num_points_roundtrip : int, optional
        If given, first apply :func:`alm_counts_roundtrip` with this
        ``num_points`` (needed for bitwise equality with the historical chain
        when ``alm`` comes straight from the kernel).
    """
    alm = np.asarray(alm)
    if l_max is None:
        l_max = int(round(np.sqrt(alm.shape[1]))) - 1
    if num_points_roundtrip is not None:
        alm = alm_counts_roundtrip(alm, num_points_roundtrip)
    return CnlmProjector(r_f, f_dict, r_al, l_max, r_interp_kind=r_interp_kind).project(alm)


def stack_cnlm_snapshots(
    snapshots: Sequence[Mapping[Tuple[int, int, int], complex]],
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """Turn a list of per-snapshot dicts into ``{(n,l,m): complex128 array}``."""
    if not snapshots:
        raise ValueError("no snapshots to stack")
    keys = list(snapshots[0].keys())
    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    for k in keys:
        out[k] = np.array([s[k] for s in snapshots], dtype=np.complex128)
    return out


def _atomic_target(path: str, overwrite: bool) -> str:
    path = os.fspath(path)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True (CLI: --force) to replace it.")
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    return path + ".tmp"


def write_cnlm_h5(
    path: str,
    t: np.ndarray,
    c_dict: Mapping[Tuple[int, int, int], np.ndarray],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> str:
    """Write ``c_nlm.h5`` in exactly the ``save_cnlm_to_h5`` layout.

    ``t`` is the time axis (Myr in all toolkit outputs), ``c_dict`` maps
    ``(n, l, m)`` to complex series of ``len(t)``.  Provenance in
    ``metadata`` becomes root attributes.  The file is written to
    ``<path>.tmp`` and renamed; an existing file is only replaced with
    ``overwrite=True``.
    """
    from .c_nlm_integrator import save_cnlm_to_h5

    tmp = _atomic_target(path, overwrite)
    save_cnlm_to_h5(dict(c_dict), np.asarray(t, dtype=np.float64), tmp, metadata=metadata)
    os.replace(tmp, path)
    return path


def write_alm_h5(
    path: str,
    alm_snapshots: Iterable[np.ndarray],
    num_points: int,
    *,
    mass_value: float = 170.3149822613627,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> str:
    """Write an a_lm file in exactly the ``alm_utils.create_sh_hdf5`` layout.

    ``/massArr`` (n_times,) filled with ``mass_value`` (the historical
    default is the paper's 170.3149822613627; the dataset is not used by any
    reader), and one group ``alm_%06d`` per snapshot (0-based position) with
    ``counts`` (constant ``num_points*2*num_points``) and ``ylm = alm*counts``.
    Provenance in ``metadata`` becomes root attributes.
    """
    import h5py
    from ._h5meta import write_attrs

    alm_list = [np.asarray(a, dtype=np.complex128) for a in alm_snapshots]
    if not alm_list:
        raise ValueError("no snapshots to write")
    shape = alm_list[0].shape
    for a in alm_list:
        if a.shape != shape:
            raise ValueError("all snapshots must have the same (n_r, n_lm) shape")
    counts = alm_counts(num_points)
    tmp = _atomic_target(path, overwrite)
    with h5py.File(tmp, "w") as f:
        f.create_dataset("massArr", data=np.full(len(alm_list), mass_value, dtype=np.float64))
        counts_data = np.full(shape, counts, dtype=np.float64)
        for idx, a in enumerate(alm_list):
            g = f.create_group(f"alm_{idx:06d}")
            g.create_dataset("counts", data=counts_data)
            g.create_dataset("ylm", data=a * counts)
        if metadata:
            write_attrs(f.attrs, metadata)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Per-snapshot a_lm checkpoints (shared by the pipeline and the inline recorder)
# ---------------------------------------------------------------------------
#: Version tag of the a_lm kernel and checkpoint format, hashed into the checkpoint fingerprint under the
#: historical key ``"toolkit"`` (0.3.0 hashed the package version there).  It changes only when the a_lm values
#: or the checkpoint format change, so checkpoints stay usable across toolkit releases (0.3.0 checkpoints keep
#: their hash).
KERNEL_FORMAT_VERSION = "0.3.0"


def kernel_fingerprint(
    *,
    toolkit_version: Optional[str] = None,
    input_prefix: str,
    dx: float,
    rs_side: float,
    rs_resol: int,
    length_units: str,
    r_values: np.ndarray,
    l_max: int,
    num_points: int,
    angular_grid: str,
) -> Tuple[Dict[str, Any], str]:
    """Parameters that determine an a_lm checkpoint, and their 16-hex-digit hash.

    The same dictionary is written by ``uldm-cnlm`` (``pipeline_params.json``) and
    by the PyUL inline recorder, so either can reuse the other's checkpoints.
    ``toolkit_version`` is accepted for backward compatibility and not hashed: the
    ``"toolkit"`` entry is :data:`KERNEL_FORMAT_VERSION`, so a toolkit release that
    does not change the kernel keeps every checkpoint usable.
    """
    r_values = np.ascontiguousarray(np.asarray(r_values, dtype=np.float64))
    fingerprint = {
        "toolkit": KERNEL_FORMAT_VERSION, "input_prefix": str(input_prefix), "dx": float(dx),
        "rs_side": float(rs_side), "rs_resol": int(rs_resol), "length_units": str(length_units),
        "r_values_sha256": hashlib.sha256(r_values.tobytes()).hexdigest(),
        "n_r": int(r_values.size), "l_max": int(l_max), "num_points": int(num_points),
        "angular_grid": str(angular_grid), "center": "com",
    }
    fp_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    return fingerprint, fp_hash


def checkpoint_path(ckpt_dir: str, index: int) -> str:
    """``<ckpt_dir>/ALM_#NNN.npz`` (PyUL's ``{index:03d}`` numbering)."""
    return os.path.join(os.fspath(ckpt_dir), f"ALM_#{int(index):03d}.npz")


def write_alm_checkpoint(
    path: str,
    alm: np.ndarray,
    center: np.ndarray,
    fingerprint_hash: str,
    snapshot: int,
    source_id: str,
    t_myr: Optional[float] = None,
) -> str:
    """Write one checkpoint atomically (``<path>.tmp.npz`` + ``os.replace``).

    Keys: ``alm`` (n_r, (l_max+1)**2) complex128, ``center`` (3,), ``fingerprint``,
    ``snapshot``, ``source_id`` (``'<size>:<mtime_ns>'`` of the P3R file, or
    :data:`INLINE_SOURCE_ID`) and, for inline checkpoints, ``t_myr``.
    """
    extra = {} if t_myr is None else {"t_myr": np.array(float(t_myr))}
    tmp = path + ".tmp.npz"
    np.savez(tmp, alm=np.asarray(alm, dtype=np.complex128), center=np.asarray(center, dtype=np.float64),
             fingerprint=np.array(fingerprint_hash), snapshot=np.array(int(snapshot)),
             source_id=np.array(source_id), **extra)
    os.replace(tmp, path)
    return path


def read_alm_checkpoint(path: str) -> Dict[str, Any]:
    """Read a checkpoint written by :func:`write_alm_checkpoint` (arrays copied out of the file)."""
    with np.load(path) as z:
        out = {"alm": np.array(z["alm"]), "center": np.array(z["center"]),
               "fingerprint": str(z["fingerprint"]), "snapshot": int(z["snapshot"]),
               "source_id": str(z["source_id"])}
        if "t_myr" in z.files:
            out["t_myr"] = float(z["t_myr"])
    return out
