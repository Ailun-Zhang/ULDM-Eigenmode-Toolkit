r"""
Computing time‑dependent ULDM spectral coefficients ``c_{nlm}(t)``.

This module provides a set of functions for taking the radial
eigenfunctions ``f_{n\ell}(r)`` produced by the ULDM eigenvalue solver
(``uldm_eig1.py``) together with a set of time‑series spherical harmonic
coefficients ``a_{\ell m}(r,t)`` stored in an HDF5 container and
constructing the fully separated mode coefficients ``c_{nlm}(t)``.  The
quantity ``c_{nlm}(t)`` is defined by the radial integral

    c_{nlm}(t) = ∫₀^{r_max} r² f_{nl}(r)^* a_{lm}(r,t) dr,

where ``f_{nl}(r)`` is real valued (hence the complex conjugate is
superfluous) and ``a_{lm}(r,t)`` is generally complex.  This operation
extracts the projection of the three‑dimensional ULDM wavefunction onto
the basis of radial eigenmodes times spherical harmonics, yielding a
set of time dependent coefficients that depend only on the indices
``n``, ``ℓ`` and ``m``.

Two HDF5 inputs are expected:

* The *eigenfunction* file created by ``save_eigs_to_h5`` in
  ``uldm_eig1.py``.  This file has a group ``/ell/<ℓ>`` for each
  angular momentum number, containing datasets ``r`` (the radial grid,
  uniform and identical for all ``ℓ``), ``E`` (eigenvalues) and ``f``
  (a 2D array of eigenfunctions; columns correspond to the radial
  quantum number ``n`` with zero‑based indexing).

* The *spherical harmonic* file created by ``create_sh_hdf5`` in
  ``alm_utils.py`` (or by ``cnlm_snapshot.write_alm_h5``, same layout).
  This file has a dataset ``massArr`` at the top level
  and one group per time step named ``alm_<tttttt>`` where ``tttttt``
  is a zero‑padded integer starting from 0.  Inside each group
  ``ylm`` is a two‑dimensional complex array of shape
  ``(n_r, (l_max+1)**2)`` containing the spherical harmonic
  coefficients at that time and radial position.  A companion dataset
  ``counts`` of the same shape stores a normalisation factor; the
  physical coefficients are obtained by dividing ``ylm`` by ``counts``.

Because the radial discretisations in the two files differ
(the eigenfunctions are tabulated on a high‑resolution uniform grid
extending to ``r_max`` and the spherical harmonic coefficients live on
a coarser grid with potentially different maximum radius), this module
performs interpolation of ``a_{lm}(r,t)`` onto the eigenfunction grid
prior to integration.  Cubic spline interpolation is used by default
but can be changed via the ``r_interp_kind`` argument.  For time
interpolation the user may supply a desired time grid; otherwise the
original sampling of the spherical harmonic file is used.

The resulting ``c_{nlm}(t)`` arrays are stored in an HDF5 file with
a hierarchical layout ``/n/<n>/l/<ℓ>/m_<m>`` (complex128 series) plus a
root dataset ``t``.  Utilities are provided to load and save this data
structure.

Since version 0.3.0 the a_lm and c_nlm files written by this toolkit may
carry provenance information as *root attributes* (radial grid, times,
units, snapshot indices, kernel parameters, software versions).  Attributes
are invisible to readers that iterate over groups and datasets, so the
historical layout is unchanged.

Limitations
-----------
At present these functions require the ``h5py`` library in order to
read and write HDF5 files.  If ``h5py`` is not installed then an
``ImportError`` will be raised when attempting to read the input
files.  Users running in restricted environments should install
``h5py`` or convert their data to an alternate format.

Example usage
-------------

>>> from Eig_ULDM_packages.c_nlm_integrator import (
...     load_eigenfunctions, load_a_lm, compute_cnlm,
...     save_cnlm_to_h5, load_cnlm_from_h5)
>>> r_f, f_dict = load_eigenfunctions('eigs_mass_170_Eig_25_lmax_10_nig_5120_py2.h5')
>>> alm = load_a_lm('PyUL_psi_alm.h5', r_al=r_al_code_units, final_time=3000.0)
>>> t_out, c_dict = compute_cnlm(r_f, f_dict, alm['r'], alm['t'],
...                              alm['a_lm'], r_interp_kind='cubic')
>>> save_cnlm_to_h5(c_dict, t_out, 'c_nlm.h5')

The companion notebook ``compute_c_nlm.ipynb`` demonstrates these
functions on realistic data and generates plots of |c_{nlm}(t)|².
"""

from __future__ import annotations

import os
import re
import sys
import json
import warnings
from typing import Dict, Tuple, Any, Optional, List, Iterable

import numpy as np
import scipy.interpolate as _interp
import scipy.integrate as _integrate

from ._compat import mp_context as _mp_context, trapezoid as _trapezoid

try:
    import h5py  # type: ignore[import-not-found]
except ImportError:
    h5py = None  # type: ignore[assignment]

# Global storage for multiprocessing worker.  When compute_cnlm is invoked
# with num_workers > 1, it populates this dictionary with the common
# arrays and dictionaries needed by the worker function.  The keys
# correspond to items used in _cnlm_worker below.
_cnlm_globals: Dict[str, Any] = {}


def _cnlm_worker(task: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], np.ndarray, str]:
    """Helper function executed by worker processes to compute c_{nlm}(t).

    This function is designed to be picklable so that it can be
    executed by ``multiprocessing.Pool``.  It relies on a global
    dictionary ``_cnlm_globals`` set by ``compute_cnlm`` to access
    shared data such as the integration grid, interpolation settings,
    eigenfunctions and spherical harmonic coefficients.  The task
    argument is a tuple ``(ell, n_idx, m_val)`` identifying the mode
    to compute.  The return value is a tuple containing the key
    ``(n_idx, ell, m_val)`` and the complex array of time‑dependent
    coefficients of length ``n_times``.
    """
    ell, n_idx, m_val = task
    # Retrieve common arrays from the global dictionary
    r_int = _cnlm_globals['r_int']       # 1D np.ndarray
    weight = _cnlm_globals['weight']     # 1D np.ndarray of r^2
    dr_int = _cnlm_globals['dr_int']     # scalar spacing
    n_times = _cnlm_globals['n_times']   # int
    r_al_arr = _cnlm_globals['r_al_arr'] # 1D np.ndarray
    r_interp_kind = _cnlm_globals['r_interp_kind']  # str
    f_on_grid = _cnlm_globals['f_on_grid']         # dict {(ell,n): array}
    a_lm = _cnlm_globals['a_lm']         # dict {(ell,m): 2D array}
    # Retrieve f on integration grid
    f_vec_int = f_on_grid[(ell, n_idx)]
    alm_mat = a_lm[(ell, m_val)]
    c_arr_local = np.zeros(n_times, dtype=np.complex128)
    # Loop over time indices
    for jj in range(n_times):
        a_slice = alm_mat[:, jj]
        # Determine interpolation kind for this slice; degrade order if too few points
        kind_local = r_interp_kind
        if kind_local in ("cubic", "quadratic"):
            n_ra = len(r_al_arr)
            if kind_local == "cubic" and n_ra < 4:
                kind_local = "linear"
            elif kind_local == "quadratic" and n_ra < 3:
                kind_local = "linear"
        interp_func_local = _interp.interp1d(
            r_al_arr,
            a_slice,
            kind=kind_local,
            bounds_error=False,
            fill_value=(a_slice[0], a_slice[-1]),
            assume_sorted=True,
        )
        a_interp_local = interp_func_local(r_int)
        integrand_local = weight * f_vec_int * a_interp_local
        c_arr_local[jj] = _trapezoid(integrand_local, dx=dr_int)
    # Identify worker (process) handling this chunk
    try:
        import multiprocessing as _mp
        worker_name = _mp.current_process().name
    except Exception:
        worker_name = "worker"
    return (n_idx, ell, m_val), c_arr_local, worker_name


def _init_cnlm_worker(
    r_int: np.ndarray,
    weight: np.ndarray,
    dr_int: float,
    n_times: int,
    r_al_arr: np.ndarray,
    r_interp_kind: str,
    f_on_grid: Dict[Tuple[int, int], np.ndarray],
    a_lm: Dict[Tuple[int, int], np.ndarray],
) -> None:
    """Initialise global variables for c_{nlm} worker processes.

    This function is executed in each worker process to populate the
    module‑level ``_cnlm_globals`` dictionary with the data needed to
    compute the integrals.  On platforms where the multiprocessing
    start method is ``spawn`` (e.g. Windows, Mac), global state in
    child processes is not inherited from the parent, so we must set
    these variables explicitly via the pool initializer.  See
    ``compute_cnlm`` for usage.

    Parameters
    ----------
    r_int : np.ndarray
        The integration radial grid for c_{nlm}.
    weight : np.ndarray
        The integration weight ``r_int^2``.
    dr_int : float
        The uniform spacing of ``r_int``.
    n_times : int
        Number of time points in the a_{lm} data.
    r_al_arr : np.ndarray
        Radial grid of the spherical harmonic data.
    r_interp_kind : str
        Interpolation kind for radial interpolation.
    f_on_grid : dict
        Mapping from ``(ℓ, n)`` to eigenfunctions sampled on ``r_int``.
    a_lm : dict
        Mapping from ``(ℓ, m)`` to a_{lm}(r,t) arrays on ``r_al_arr``.
    """
    _cnlm_globals['r_int'] = r_int
    _cnlm_globals['weight'] = weight
    _cnlm_globals['dr_int'] = dr_int
    _cnlm_globals['n_times'] = n_times
    _cnlm_globals['r_al_arr'] = r_al_arr
    _cnlm_globals['r_interp_kind'] = r_interp_kind
    _cnlm_globals['f_on_grid'] = f_on_grid
    _cnlm_globals['a_lm'] = a_lm


def _init_cnlm_worker_lm(
    r_int: np.ndarray,
    weight: np.ndarray,
    dr_int: float,
    n_times: int,
    r_al_arr: np.ndarray,
    r_interp_kind: str,
    f_on_grid: Dict[Tuple[int, int], np.ndarray],
) -> None:
    """Initialiser for :func:`_cnlm_worker_lm` (everything except a_lm)."""
    _init_cnlm_worker(r_int, weight, dr_int, n_times, r_al_arr, r_interp_kind, f_on_grid, {})


def _cnlm_worker_lm(task):
    """Compute c_{nlm}(t) for one (ℓ, m) and all requested n.

    ``task = (ell, m_val, n_list, alm_mat)``.  Only the a_lm matrix of this
    (ℓ, m) is sent to the worker (instead of pickling the whole a_lm
    dictionary into every worker), and the radial interpolation of each time
    slice is computed once and reused for all n.  The arithmetic per
    (n, ℓ, m, t) is identical to :func:`_cnlm_worker`, so results are
    bitwise equal.
    """
    ell, m_val, n_list, alm_mat = task
    g = _cnlm_globals
    r_int = g['r_int']; weight = g['weight']; dr_int = g['dr_int']
    n_times = g['n_times']; r_al_arr = g['r_al_arr']; r_interp_kind = g['r_interp_kind']
    f_on_grid = g['f_on_grid']
    out = {n_idx: np.zeros(n_times, dtype=np.complex128) for n_idx in n_list}
    for jj in range(n_times):
        a_slice = alm_mat[:, jj]
        kind_local = r_interp_kind
        if kind_local in ("cubic", "quadratic"):
            n_ra = len(r_al_arr)
            if kind_local == "cubic" and n_ra < 4:
                kind_local = "linear"
            elif kind_local == "quadratic" and n_ra < 3:
                kind_local = "linear"
        interp_func_local = _interp.interp1d(
            r_al_arr,
            a_slice,
            kind=kind_local,
            bounds_error=False,
            fill_value=(a_slice[0], a_slice[-1]),
            assume_sorted=True,
        )
        a_interp_local = interp_func_local(r_int)
        for n_idx in n_list:
            integrand_local = weight * f_on_grid[(ell, n_idx)] * a_interp_local
            out[n_idx][jj] = _trapezoid(integrand_local, dx=dr_int)
    try:
        import multiprocessing as _mp
        worker_name = _mp.current_process().name
    except Exception:
        worker_name = "worker"
    return [((n_idx, ell, m_val), out[n_idx]) for n_idx in n_list], worker_name


def check_rmax_consistency(
    r_al_max: Any,
    r_max_eig: Any,
    *,
    rtol: float = 1e-6,
    raise_error: bool = True,
    label: str = "a_lm radial grid",
) -> float:
    """Check that the a_lm shell grid and the eigenfunction grid end at the same radius.

    The historical ``compute_c_nlm.ipynb`` silently rescaled the a_lm radii
    onto the eigen-grid r_max (``r_al = r/r[-1]*r_max_eig``).  That rescale
    is harmless only when both grids already end at the same physical
    radius (true for the paper: 2.5 kpc = 0.20604798941694438 code units at
    m_a = 1e-21 eV).  Call this helper first.

    Parameters
    ----------
    r_al_max, r_max_eig : float or 1-D array
        Outer radii (or whole grids; the last element is used), in the
        same units (code units for compute_cnlm).
    rtol : float
        Allowed relative mismatch.
    raise_error : bool
        Raise ``ValueError`` on mismatch (default) or only warn.

    Returns
    -------
    float
        The relative mismatch ``|r_al_max - r_max_eig| / r_max_eig``.
    """
    a = float(np.asarray(r_al_max, dtype=np.float64).ravel()[-1])
    b = float(np.asarray(r_max_eig, dtype=np.float64).ravel()[-1])
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        raise ValueError(f"invalid radii for r_max check: {a!r}, {b!r}")
    rel = abs(a - b) / b
    if rel > rtol:
        msg = (
            f"{label} ends at r = {a:.15g} but the eigenfunction grid ends at r_max = {b:.15g} "
            f"(relative mismatch {rel:.3e} > rtol={rtol:g}). Rescaling one onto the other would "
            "silently stretch all radii; check the RS box length, the length units and the axion mass."
        )
        if raise_error:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return rel


def rescale_r_to_eigen_rmax(
    r_values: np.ndarray,
    r_max_eig: float,
    *,
    rtol: float = 1e-6,
) -> np.ndarray:
    """``(r_values / r_values[-1]) * r_max_eig`` after :func:`check_rmax_consistency`.

    This reproduces the radial-grid line of ``compute_c_nlm.ipynb``
    bit for bit, but refuses to stretch the grid when the two outer radii
    disagree by more than ``rtol``.
    """
    r_values = np.asarray(r_values, dtype=np.float64)
    check_rmax_consistency(r_values, r_max_eig, rtol=rtol)
    return (r_values / r_values[-1]) * r_max_eig


def load_eigenfunctions(
    eigs_path: str,
    l_subset: Optional[Iterable[int]] = None,
    n_subset: Optional[Dict[int, Iterable[int]]] = None,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    r"""Load radial eigenfunctions ``f_{n\ell}(r)`` from an HDF5 file.

    Parameters
    ----------
    eigs_path : str
        Path to the eigenfunction HDF5 file.  This file must follow the
        schema produced by ``save_eigs_to_h5`` in ``uldm_eig1.py``.

    l_subset : iterable of int, optional
        If provided, only these angular momentum values will be loaded.
        By default all ``ℓ`` present in the file are loaded.  Values
        outside the range of available groups are ignored.

    n_subset : dict mapping int to iterable of int, optional
        Optionally restrict the radial quantum numbers ``n`` loaded for
        each ``ℓ``.  The dictionary keys are angular momentum values
        and the values are iterables specifying which ``n`` indices to
        include.  Missing keys imply that all ``n`` for that ``ℓ``
        should be loaded.  Negative indices or indices beyond the
        number of eigenvalues are silently ignored.

    Returns
    -------
    r_f : np.ndarray
        One‑dimensional array of the radial grid on which the
        eigenfunctions are tabulated.  This grid is uniform and
        identical for all ``ℓ``.  Its length is ``n_r``.

    f_dict : dict
        Mapping from ``(ℓ, n)`` to a one‑dimensional array of shape
        ``(n_r,)`` containing ``f_{nℓ}(r)``.  Only those pairs
        requested by ``l_subset`` and ``n_subset`` are included.

    Raises
    ------
    ImportError
        If ``h5py`` is unavailable.

    KeyError
        If the file does not contain the expected groups or datasets.

    ValueError
        If the radial grids differ between ℓ channels.
    """
    if h5py is None:
        raise ImportError(
            "h5py is required to read eigenfunction data but is not installed."
        )
    f_dict: Dict[Tuple[int, int], np.ndarray] = {}
    r_f: Optional[np.ndarray] = None
    with h5py.File(eigs_path, 'r') as f:
        if 'ell' not in f:
            raise KeyError(f"Group 'ell' not found in {eigs_path}.")
        ell_group = f['ell']
        # Determine which ℓ to process
        keys = list(ell_group.keys())
        available_ls = sorted([int(k) for k in keys if k.isdigit()])
        ls_to_load: List[int]
        if l_subset is None:
            ls_to_load = available_ls
        else:
            ls_to_load = [l for l in l_subset if l in available_ls]
        for ell in ls_to_load:
            g = ell_group[str(ell)]
            if 'r' not in g or 'f' not in g:
                raise KeyError(
                    f"Group 'ell/{ell}' in {eigs_path} lacks 'r' or 'f' datasets."
                )
            r_vals = np.asarray(g['r'], dtype=np.float64)
            if r_f is None:
                r_f = r_vals
            else:
                if r_f.shape != r_vals.shape or not np.allclose(r_f, r_vals, rtol=1e-12, atol=1e-12):
                    raise ValueError(
                        f"Radial grid mismatch in ell/{ell}: expected {r_f.shape}, got {r_vals.shape}."
                    )
            # The dataset f contains real eigenfunctions stored as float64.
            # Reading directly into a complex dtype can trigger a HDF5 read
            # error ('no appropriate function for conversion path'), so read
            # the dataset as its native dtype (float64) and convert to
            # complex only when needed.  h5py will then perform a simple
            # memory copy instead of attempting an unsupported conversion.
            f_mat = np.asarray(g['f'], dtype=np.float64)
            # Determine which n to load for this ℓ
            n_indices: Iterable[int]
            if n_subset is None or ell not in n_subset:
                n_indices = range(f_mat.shape[1])
            else:
                n_indices = [ni for ni in n_subset[ell] if 0 <= ni < f_mat.shape[1]]
            for n_idx in n_indices:
                f_dict[(ell, n_idx)] = f_mat[:, n_idx].astype(np.complex128)
    if r_f is None:
        raise KeyError(f"No eigenfunction data found in {eigs_path}.")
    return r_f, f_dict


_RMAX_TOKEN_RE = re.compile(
    r"(?:^|[_\-.(])r_?max[_=\-]?([0-9]+(?:\.[0-9]*)?(?:[eE][+\-]?[0-9]+)?)(?=$|[_\-()]|\.h5$|\.hdf5$)",
    re.IGNORECASE,
)


def _parse_rmax_from_filename(filename: str) -> Optional[float]:
    """Extract an *explicit* maximum radius token from an a_lm file name.

    Only an explicit ``rmax`` token is recognised, e.g.
    ``PyUL_psi_alm_rmax0.206_mass170.h5``, ``..._rmax_0.206_...`` or
    ``..._r_max=0.206.h5`` (case-insensitive).  The value must be in the
    units of the eigenfunction grid (code units).

    Historical note: before version 0.3.0 the number after ``psi_alm_`` was
    taken as r_max.  The paper's own file name
    ``PyUL_psi_alm_0.02_mass170_256.h5`` would then have been read as
    r_max = 0.02 (that number is the BH mass ratio), so that convention is no
    longer used; such names return ``None`` and :func:`load_a_lm` asks for an
    explicit ``r_al``.

    Parameters
    ----------
    filename : str
        Name or path of the HDF5 file.

    Returns
    -------
    float or None
        The parsed radius, or ``None`` when the name has no explicit token.
    """
    base = os.path.basename(str(filename))
    match = _RMAX_TOKEN_RE.search(base)
    if match is None:
        return None
    try:
        val = float(match.group(1))
    except ValueError:
        return None
    return val if val > 0 else None


def load_a_lm(
    alm_path: str,
    r_al: np.ndarray,
    r_max_override: Optional[float] = None,
    final_time: float = 1.0,
    l_max: Optional[int] = None,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    verbose: bool = False,
    *,
    times: Optional[np.ndarray] = None,
    snapshot_indices: Optional[Iterable[int]] = None,
    save_interval: Optional[float] = None,
    time_source: str = 'auto',
) -> Dict[str, Any]:
    r"""Load spherical harmonic coefficients ``a_{ℓm}(r,t)`` from an HDF5 file.

    This function opens the HDF5 file created by ``create_sh_hdf5`` and
    reconstructs the complex spherical harmonic coefficients at each
    radial point and time.  The coefficients are stored internally as
    ``ylm`` divided by ``counts``.  The returned dictionary contains
    the radial grid, the time grid and a mapping from ``(ℓ,m)`` to a
    two‑dimensional complex array of shape ``(n_r, n_times)``.

    If a radial grid ``r_al`` is not supplied, the function attempts
    to infer one, in this order: ``r_max_override``; a dataset named
    ``r`` in the file; the root attribute ``r_values_code`` (written by
    toolkit >= 0.3.0, code units, i.e. the units of the eigenfunction
    grid); the root attribute ``r_values`` when its ``r_units`` attribute
    is ``''`` (code units); and finally an explicit ``rmax`` token in the
    file name (see :func:`_parse_rmax_from_filename`).  The ``r_max`` based
    variants build a uniform grid of ``n_r`` points with the first element
    set to ``1e-5``.

    Time axis (``time_source``)
    ---------------------------
    * ``'auto'`` (default): explicit ``times`` > explicit
      ``snapshot_indices`` times ``save_interval`` > the root attribute
      ``times`` of files written by toolkit >= 0.3.0 > the historical
      default ``linspace(0, final_time, n_times)``.  For files without
      time attributes (all historical files) and no explicit arguments the
      result is therefore unchanged.
    * ``'legacy'``: always ``linspace(0, final_time, n_times)``.
    * ``'file'``: require the ``times`` attribute in the file.

    ``n_times`` is the number of ``alm_*`` groups in the file, optionally
    restricted by ``start_index`` and ``end_index`` (positions of the
    groups in the file, not PyUL snapshot numbers).  Note that the legacy
    time axis always starts at 0 even when the file starts at a later
    snapshot; use ``snapshot_indices``/``save_interval`` or ``times`` (or a
    file with time attributes) to get the true times.

    Parameters
    ----------
    alm_path : str
        Path to the HDF5 file containing ``ylm`` and ``counts`` data.

    r_al : np.ndarray
        Explicit radial grid on which the spherical harmonic data are
        tabulated.  This array must be one‑dimensional with length
        equal to the radial dimension of the ``ylm`` dataset.  If
        ``None`` and ``r_max_override`` is also ``None`` the grid
        cannot be inferred and a ``ValueError`` is raised.  Passing
        this argument explicitly is recommended to avoid ambiguities.

    final_time : float, optional
        Physical time at the last index.  The returned time grid spans
        ``[0, final_time]``.  Defaults to 1.0.

    l_max : int, optional
        If supplied, only spherical harmonic indices up to and
        including ``l_max`` are loaded.  Otherwise ``l_max`` is
        determined from the shape of the ``ylm`` dataset.

    start_index, end_index : int, optional
        Restrict the time steps to load.  These refer to the zero‑based
        ordering of ``alm_*`` groups in the file.  If ``None`` the
        range spans all available groups.  Negative values or values
        outside the available range are clipped silently.

    verbose : bool, optional
        If True print diagnostic messages when inferring grids.

    times : array-like, optional
        Explicit times of the selected groups (length ``n_times``).

    snapshot_indices : iterable of int, optional
        PyUL snapshot numbers of the selected groups; with
        ``save_interval`` (time between PyUL saves, i.e.
        ``duration/save_number``) the times are
        ``snapshot_indices * save_interval``.

    save_interval : float, optional
        See ``snapshot_indices``.

    time_source : {'auto', 'legacy', 'file'}
        See "Time axis" above.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``'r'``: one‑dimensional array of length ``n_r`` representing
          the radial grid used for the spherical harmonic data.
        - ``'t'``: one‑dimensional array of length ``n_times``
          containing the time values in physical units.
        - ``'a_lm'``: mapping from ``(ℓ,m)`` to a complex array of
          shape ``(n_r, n_times)`` storing ``a_{ℓm}(r,t)``.
        - ``'snapshot_indices'``: PyUL snapshot numbers of the loaded
          groups when known (explicit argument or file attribute), else
          ``None``.
        - ``'time_source'``: which rule produced ``'t'``.
        - ``'attrs'``: the root attributes of the file (empty for
          historical files).

    Raises
    ------
    ImportError
        If ``h5py`` is not available.

    KeyError
        If the HDF5 structure is missing expected groups or datasets.

    ValueError
        If the inferred radial grid cannot be determined.
    """
    if h5py is None:
        raise ImportError(
            "h5py is required to read spherical harmonic data but is not installed."
        )
    from ._h5meta import read_array_attr, attrs_to_dict

    if time_source not in ('auto', 'legacy', 'file'):
        raise ValueError("time_source must be one of 'auto', 'legacy', 'file'.")
    # Open the file and enumerate groups
    with h5py.File(alm_path, 'r') as f:
        file_attrs = attrs_to_dict(f.attrs)
        # Identify time groups (alm_000000, alm_000001, ...)
        group_names = [name for name in f.keys() if name.startswith('alm_')]
        if not group_names:
            raise KeyError(f"No groups named 'alm_*' found in {alm_path}.")
        # Sort lexicographically which corresponds to increasing index due to zero‑padding
        group_names.sort()
        # Apply start/end index restrictions
        total_groups = len(group_names)
        s_idx = 0 if start_index is None else max(0, int(start_index))
        e_idx = total_groups - 1 if end_index is None else min(total_groups - 1, int(end_index))
        if s_idx > e_idx:
            raise ValueError(
                f"start_index {start_index} must be <= end_index {end_index}."
            )
        sel_group_names = group_names[s_idx:e_idx + 1]
        n_times = len(sel_group_names)
        # Determine n_r and n_lm from the first selected group
        first_grp = f[sel_group_names[0]]
        if 'ylm' not in first_grp or 'counts' not in first_grp:
            raise KeyError(
                f"Group '{sel_group_names[0]}' lacks 'ylm' or 'counts' datasets."
            )
        ylm0 = np.asarray(first_grp['ylm'], dtype=np.complex128)
        counts0 = np.asarray(first_grp['counts'], dtype=np.float64)
        if ylm0.shape != counts0.shape:
            raise ValueError(
                f"In group '{sel_group_names[0]}', 'ylm' and 'counts' shapes differ:"
                f" {ylm0.shape} vs {counts0.shape}."
            )
        n_r, n_lm_total = ylm0.shape
        # Determine l_max from n_lm_total if not provided
        inferred_lmax = int(np.sqrt(n_lm_total) - 1)
        if l_max is None:
            l_max_use = inferred_lmax
        else:
            l_max_use = min(int(l_max), inferred_lmax)
        # Build mapping from linear index to (l,m)
        # Use the same ordering as reorganize_coeffs and get_lm_label from a_lm.py
        lm_list: List[Tuple[int, int]] = []
        for l in range(l_max_use + 1):
            for m in range(-l, l + 1):
                lm_list.append((l, m))
        n_lm_use = len(lm_list)
        # Prepare storage for a_{lm}(r,t)
        # We'll allocate a dict mapping (l,m) to an array (n_r, n_times)
        a_lm: Dict[Tuple[int, int], np.ndarray] = {
            (l, m): np.zeros((n_r, n_times), dtype=np.complex128) for l, m in lm_list
        }
        # Determine the radial grid r_al
        # Priority: explicit r_al argument > r_max_override > r dataset in file > parse from filename
        if r_al is not None:
            # User supplied the full radial grid
            r_al_arr = np.asarray(r_al, dtype=np.float64)
            if r_al_arr.ndim != 1 or r_al_arr.size != n_r:
                raise ValueError(
                    f"Provided r_al has shape {r_al_arr.shape}, expected ({n_r},)."
                )
            r_al_final = r_al_arr
        elif r_max_override is not None:
            # Construct a uniform grid from 0 to r_max_override inclusive
            # Replace the first element with 1e-5 to avoid singularity
            r_max_override = float(r_max_override)
            if r_max_override <= 0:
                raise ValueError("r_max_override must be positive.")
            r_tmp = np.linspace(0.0, r_max_override, n_r, endpoint=True, dtype=np.float64)
            r_tmp[0] = 1e-5
            r_al_final = r_tmp
            if verbose:
                print(f"Using override radial grid with r_max = {r_al_final[-1]:.6g}")
        else:
            # Try to read r grid from file if present
            r_al_use: Optional[np.ndarray] = None
            if 'r' in f:
                candidate = np.asarray(f['r'], dtype=np.float64)
                if candidate.ndim == 1 and candidate.size == n_r:
                    r_al_use = candidate
            if r_al_use is None:
                # Provenance attributes written by toolkit >= 0.3.0
                cand = read_array_attr(f.attrs, 'r_values_code')
                if cand is None and str(file_attrs.get('r_units', 'unknown')) == '':
                    cand = read_array_attr(f.attrs, 'r_values')
                if cand is not None:
                    cand = np.asarray(cand, dtype=np.float64)
                    if cand.ndim == 1 and cand.size == n_r:
                        r_al_use = cand
                        if verbose:
                            print("Using radial grid from the file attributes (code units).")
            if r_al_use is None:
                # Attempt to parse an explicit rmax token from the filename
                rmax = _parse_rmax_from_filename(os.path.basename(alm_path))
                if rmax is None:
                    hint = ''
                    if read_array_attr(f.attrs, 'r_values') is not None:
                        hint = (f" The file stores 'r_values' in units {file_attrs.get('r_units')!r}; convert"
                                " them to code units (the units of the eigenfunction grid) and pass r_al.")
                    raise ValueError(
                        "Cannot infer radial grid; provide r_al or r_max_override, or store an 'r' dataset."
                        + hint
                    )
                # Uniform grid between r=0 and r=rmax inclusive.  First point replaced by 1e-5
                r_tmp = np.linspace(0.0, rmax, n_r, endpoint=True, dtype=np.float64)
                r_tmp[0] = 1e-5
                r_al_use = r_tmp
            r_al_final = r_al_use
            if verbose:
                print(f"Inferred radial grid of length {n_r} with r_max = {r_al_final[-1]:.6g}")
        # Snapshot indices of the selected groups, when known
        sel_snap: Optional[np.ndarray] = None
        file_snap = read_array_attr(f.attrs, 'snapshot_indices')
        if snapshot_indices is not None:
            sel_snap = np.asarray(list(snapshot_indices), dtype=np.int64)
            if sel_snap.shape != (n_times,):
                raise ValueError(
                    f"snapshot_indices has length {sel_snap.size}, expected {n_times} (selected groups)."
                )
        elif file_snap is not None and np.size(file_snap) == total_groups:
            sel_snap = np.asarray(file_snap, dtype=np.int64)[s_idx:e_idx + 1]
        # Construct the time vector
        file_times = read_array_attr(f.attrs, 'times')
        if file_times is not None and np.size(file_times) != total_groups:
            file_times = None
        if time_source == 'legacy':
            t_used = 'legacy linspace(0, final_time, n_times)'
            t_array = np.linspace(0.0, float(final_time), n_times, endpoint=True, dtype=np.float64)
        elif times is not None:
            t_array = np.asarray(times, dtype=np.float64)
            if t_array.shape != (n_times,):
                raise ValueError(f"times has shape {t_array.shape}, expected ({n_times},).")
            t_used = 'explicit times'
        elif snapshot_indices is not None and save_interval is not None:
            t_array = sel_snap.astype(np.float64) * float(save_interval)
            t_used = 'snapshot_indices * save_interval'
        elif file_times is not None:
            t_array = np.asarray(file_times, dtype=np.float64)[s_idx:e_idx + 1]
            t_used = 'file attribute times'
            if final_time != 1.0 and verbose:
                print(f"Using the time axis stored in the file (final_time={final_time} ignored).")
        elif time_source == 'file':
            raise ValueError(f"time_source='file' but {alm_path} has no usable 'times' attribute.")
        else:
            t_used = 'legacy linspace(0, final_time, n_times)'
            # Construct time vector from 0 to final_time inclusive
            t_array = np.linspace(0.0, float(final_time), n_times, endpoint=True, dtype=np.float64)
        # Loop over selected time groups and populate a_lm arrays
        for t_idx, gname in enumerate(sel_group_names):
            grp = f[gname]
            ylm = np.asarray(grp['ylm'], dtype=np.complex128)
            counts = np.asarray(grp['counts'], dtype=np.float64)
            if ylm.shape[0] != n_r or ylm.shape[1] != n_lm_total:
                raise ValueError(
                    f"Group '{gname}' has unexpected shape {ylm.shape}; expected ({n_r}, {n_lm_total})."
                )
            # Compute a_lm = ylm / counts.  Avoid division by zero
            # counts is typically constant across l,m but we do elementwise division
            # Use where to handle any zeros gracefully
            with np.errstate(divide='ignore', invalid='ignore'):
                alm_vals = np.where(counts != 0, ylm / counts, 0.0)
            # Reorganise columns into (l,m).  Only take the first n_lm_use columns
            for idx, (l, m) in enumerate(lm_list):
                a_lm[(l, m)][:, t_idx] = alm_vals[:, idx]
        return {
            'r': r_al_final,
            't': t_array,
            'a_lm': a_lm,
            'snapshot_indices': sel_snap,
            'time_source': t_used,
            'attrs': file_attrs,
        }


def compute_cnlm(
    r_f: np.ndarray,
    f_dict: Dict[Tuple[int, int], np.ndarray],
    r_al: np.ndarray,
    t_al: np.ndarray,
    a_lm: Dict[Tuple[int, int], np.ndarray],
    *,
    r_interp_kind: str = 'cubic',
    time_interp: Optional[str] = None,
    t_out: Optional[np.ndarray] = None,
    grid_choice: str = 'eigen',
    r_manual: Optional[np.ndarray] = None,
    num_workers: int = 1,
    progress: bool = False,
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], np.ndarray]]:
    r"""Compute the spectral coefficients ``c_{n\ell m}(t)`` via radial integration.

    Given a set of radial eigenfunctions ``f_{n\ell}(r)`` on a fine
    grid ``r_f`` and spherical harmonic coefficients ``a_{\ell m}(r,t)`` on
    a (possibly different) grid ``r_al``, this function computes

    ``c_{n\ell m}(t_j) = ∫ r² f_{n\ell}(r) a_{\ell m}(r,t_j) dr``

    for each supplied time index ``j`` and each combination of
    ``(ℓ,n,m)`` available in ``f_dict`` and ``a_lm``.  The integral is
    performed using composite trapezoidal integration on the fine grid
    ``r_f``.  Because the spherical harmonic data are tabulated on
    ``r_al`` the function first interpolates ``a_{ℓm}(r,t_j)`` onto
    ``r_f``.  The interpolation method can be controlled via the
    ``r_interp_kind`` argument (options include ``'linear'``, ``'cubic'``,
    etc.).  Time interpolation may also be performed if a new time
    array ``t_out`` is provided; otherwise the coefficients are
    computed on the original time grid ``t_al``.

    Parameters
    ----------
    r_f : np.ndarray
        Fine radial grid on which the eigenfunctions ``f_{n\ell}(r)``
        are defined.  Must be one‑dimensional and strictly increasing.

    f_dict : dict
        Mapping from ``(ℓ,n)`` to arrays of shape ``(n_r_f,)`` giving
        ``f_{n\ell}(r)`` on ``r_f``.  Only those pairs present in this
        dictionary will be processed.  The keys should align with the
        angular momentum values present in ``a_lm``.

    r_al : np.ndarray
        Radial grid for ``a_{\ell m}(r,t)``.  Must be one‑dimensional
        and strictly increasing.  Typically coarser than ``r_f``.  The
        first element is assumed to be greater than zero (to avoid
        division by zero in spherical coordinates).

    t_al : np.ndarray
        Time grid corresponding to the second dimension of each
        ``a_{\ell m}`` array.

    a_lm : dict
        Mapping from ``(ℓ,m)`` to arrays of shape ``(n_r_al, n_times)``
        giving the spherical harmonic coefficients on the coarse grid.

    r_interp_kind : str, optional
        Kind of radial interpolation to use when mapping
        ``a_{\ell m}(r,t_j)`` onto ``r_f``.  Defaults to ``'cubic'``.
        See ``scipy.interpolate.interp1d`` for allowed values.  If
        ``r_f`` lies partially outside the range of ``r_al`` then
        values are extrapolated using the nearest valid value.

    time_interp : {'linear','nearest','cubic','pchip','akima','cspline-natural'}, optional
        If provided along with a new time array ``t_out``, specify the
        interpolation method used to resample the computed ``c_{nlm}(t)``
        onto ``t_out``.  Note: this resampling is performed after the
        radial integrals have been computed on the original time grid
        ``t_al`` (i.e. we never time‑interpolate ``a_{\ell m}(r,t)``
        prior to integration here). If ``None`` and ``t_out`` is
        provided, linear interpolation is used.  If ``t_out`` is
        ``None`` then no time interpolation is performed and the
        original time grid ``t_al`` is used.

    t_out : np.ndarray, optional
        Desired output time grid.  If supplied, this array must be
        one‑dimensional and strictly increasing.  The resulting
        coefficients will be evaluated at these times using
        interpolation across the pre‑computed values on ``t_al``.

    progress : bool, optional
        If ``True`` print progress messages during the computation.

    Additional grid selection options
    -------------------------------
    The radial integration may be carried out on one of several grids.
    By default ``grid_choice='eigen'`` is used, which means the
    integration uses the high‑resolution grid ``r_f`` on which the
    eigenfunctions are tabulated.  In this case the spherical
    harmonic coefficients ``a_{\ell m}(r,t)`` are interpolated onto
    ``r_f`` before integration.  This option maximises accuracy but
    can be computationally intensive if ``r_f`` has many points.

    If ``grid_choice='alm'`` the integration grid is taken to be
    ``r_al``, the coarse radial grid of the spherical harmonic data.
    Here the eigenfunctions are interpolated onto this coarser grid
    before integration.  This reduces the number of radial points
    involved in each integral at the expense of interpolation error.

    If ``grid_choice='manual'`` the user must supply ``r_manual``.
    This array defines a custom radial grid for the integration.  Both
    the eigenfunctions and the spherical harmonic coefficients are
    interpolated onto this grid prior to integration.  The manual grid
    must lie within the domain of both input grids; extrapolation is
    handled by repeating the boundary values.  The first element of
    ``r_manual`` should be non‑zero to avoid the singularity at the
    origin.

    Parallel execution
    ------------------
    The computation of ``c_{nlm}(t)`` for different combinations of
    indices can be expensive.  The optional parameter ``num_workers``
    specifies how many CPU processes to use for the radial integrals.
    A value of 1 disables multiprocessing and runs the loops serially.
    When ``num_workers`` > 1, the integrals for each triple ``(n,
    ℓ, m)`` are distributed evenly among the workers and computed in
    parallel.  Note that progress reporting is only supported in the
    serial case; when running in parallel the function returns
    silently until completion.

    Returns
    -------
    t_return : np.ndarray
        The time grid on which the returned coefficients are defined.
        This is either ``t_al`` (if no resampling is performed) or
        ``t_out``.

    c_dict : dict
        Mapping from ``(n, \ell, m)`` to complex arrays of shape
        ``(n_times_return,)``.  Each array contains the integral
        ``c_{nlm}(t)`` evaluated on the corresponding time grid.

    Raises
    ------
    ValueError
        If the input grids are inconsistent or improperly ordered.
    """
    # Validate input arrays
    r_f_arr = np.asarray(r_f, dtype=np.float64)
    r_al_arr = np.asarray(r_al, dtype=np.float64)
    if r_f_arr.ndim != 1 or r_al_arr.ndim != 1:
        raise ValueError("r_f and r_al must be one‑dimensional arrays.")
    if not np.all(np.diff(r_f_arr) > 0):
        raise ValueError("r_f must be strictly increasing.")
    if not np.all(np.diff(r_al_arr) > 0):
        raise ValueError("r_al must be strictly increasing.")
    if t_al.ndim != 1 or not np.all(np.diff(t_al) >= 0):
        raise ValueError("t_al must be a one‑dimensional nondecreasing array.")
    n_times = len(t_al)
    # Determine time grid for return
    if t_out is None:
        t_ret = np.asarray(t_al, dtype=np.float64)
        need_time_resample = False
    else:
        t_ret = np.asarray(t_out, dtype=np.float64)
        if t_ret.ndim != 1 or not np.all(np.diff(t_ret) >= 0):
            raise ValueError("t_out must be one‑dimensional and nondecreasing.")
        need_time_resample = True
        # Select time interpolation kind
        kind_time = time_interp if time_interp is not None else 'linear'
        if kind_time not in ('linear', 'nearest', 'cubic', 'pchip', 'akima', 'cspline-natural'):
            raise ValueError("time_interp must be one of 'linear','nearest','cubic','pchip','akima','cspline-natural'.")

    # Choose radial grid for integration
    grid_choice = grid_choice.lower()
    if grid_choice not in ('eigen', 'alm', 'manual'):
        raise ValueError("grid_choice must be one of 'eigen', 'alm', 'manual'.")
    if grid_choice == 'manual':
        if r_manual is None:
            raise ValueError("r_manual must be supplied when grid_choice='manual'.")
        r_int = np.asarray(r_manual, dtype=np.float64)
        if r_int.ndim != 1 or not np.all(np.diff(r_int) > 0):
            raise ValueError("r_manual must be one‑dimensional and strictly increasing.")
    elif grid_choice == 'eigen':
        r_int = r_f_arr
    else:  # 'alm'
        r_int = r_al_arr
    # Compute integration weights and spacing for trapezoidal rule on r_int
    weight = r_int ** 2
    dr_int = r_int[1] - r_int[0]
    # For grid_choice 'alm' or 'manual' we will need to interpolate f onto r_int
    # Prepare a dictionary mapping (ell,n) to f on r_int
    if grid_choice == 'eigen':
        f_on_grid: Dict[Tuple[int, int], np.ndarray] = f_dict
    else:
        # Interpolate each eigenfunction onto r_int
        f_on_grid = {}
        for (ell, n_idx), f_vec in f_dict.items():
            # Determine local interpolation kind: degrade if too few points
            kind_local = r_interp_kind
            if kind_local in ("cubic", "quadratic"):
                n_rf = len(r_f_arr)
                if kind_local == "cubic" and n_rf < 4:
                    kind_local = "linear"
                elif kind_local == "quadratic" and n_rf < 3:
                    kind_local = "linear"
            interp_func_f = _interp.interp1d(
                r_f_arr,
                f_vec,
                kind=kind_local,
                bounds_error=False,
                fill_value=(f_vec[0], f_vec[-1]),
                assume_sorted=True,
            )
            f_on_grid[(ell, n_idx)] = interp_func_f(r_int)
    # For grid_choice 'manual', we also need to interpolate a onto r_int
    # However, interpolation for a will be performed on the fly for each time step below.

    # Prepare output dictionary keyed by (n, ell, m)
    c_dict: Dict[Tuple[int, int, int], np.ndarray] = {}

    # Helper function for serial computation
    def _compute_for_mode(ell: int, n_idx: int, m_val: int) -> Tuple[Tuple[int, int, int], np.ndarray]:
        """Compute c_{nlm}(t) for a single triple in serial mode."""
        # Retrieve f on integration grid
        f_vec_int = f_on_grid[(ell, n_idx)]
        # Retrieve a_{ℓm} matrix on its native grid r_al
        alm_mat = a_lm[(ell, m_val)]  # shape (n_r_al, n_times)
        # Prepare array of integrals on native time grid
        c_arr = np.zeros(n_times, dtype=np.complex128)
        # For each time index perform radial interpolation of a and integration
        for j in range(n_times):
            a_slice = alm_mat[:, j]
            # Interpolate a onto r_int
            # Choose interpolation kind; degrade if too few points
            kind_local = r_interp_kind
            if kind_local in ("cubic", "quadratic"):
                n_ra = len(r_al_arr)
                if kind_local == "cubic" and n_ra < 4:
                    kind_local = "linear"
                elif kind_local == "quadratic" and n_ra < 3:
                    kind_local = "linear"
            interp_func = _interp.interp1d(
                r_al_arr,
                a_slice,
                kind=kind_local,
                bounds_error=False,
                fill_value=(a_slice[0], a_slice[-1]),
                assume_sorted=True,
            )
            a_interp = interp_func(r_int)
            # Compute integrand and integrate
            integrand = weight * f_vec_int * a_interp
            c_arr[j] = _trapezoid(integrand, dx=dr_int)
        return (n_idx, ell, m_val), c_arr

    # Serial computation with progress reporting
    if num_workers <= 1:
        total_jobs = sum(
            1
            for (ell, n_idx) in f_on_grid.keys()
            for m in range(-ell, ell + 1)
            if (ell, m) in a_lm
        )
        job_count = 0
        last_percent_print = -1
        # Render a single-line progress bar (works in notebooks and terminals)
        def _render_progress_serial(percent: int, done: int, total: int) -> None:
            bar_len = 30
            filled = int(bar_len * max(0, min(100, percent)) / 100)
            bar = '=' * filled + '.' * (bar_len - filled)
            msg = f"[{bar}] {percent:3d}% ({done}/{total})"
            try:
                from IPython.display import clear_output  # type: ignore
                clear_output(wait=True)
                print(msg, end='', flush=True)
            except Exception:
                sys.stdout.write('\r' + msg)
                sys.stdout.flush()
        for (ell, n_idx) in sorted(f_on_grid.keys()):
            # Identify valid m values for this ell
            m_vals = [m for (l2, m) in a_lm.keys() if l2 == ell]
            for m_val in sorted(m_vals):
                key, c_arr = _compute_for_mode(ell, n_idx, m_val)
                c_dict[key] = c_arr
                job_count += 1
                if progress:
                    percent = int((job_count * 100) / total_jobs)
                    if percent != last_percent_print:
                        _render_progress_serial(percent, job_count, total_jobs)
                        last_percent_print = percent
        if progress:
            # Ensure the bar is full and move to next line
            _render_progress_serial(100, job_count, total_jobs)
            print()
    else:
        # Parallel computation using a multiprocessing pool (start method from _compat.mp_context,
        # i.e. ULDM_MP_START_METHOD or the platform default) and a global worker
        # Build one task per (ell, m_val) carrying only that a_lm matrix and the
        # list of n for this ell.  Previously the whole a_lm dictionary was
        # pickled into every worker through the pool initializer (about
        # 250 MB per worker for the paper's 121 x 257 x 501 a_lm).  Results are
        # bitwise identical (same arithmetic per (n, ell, m, t)).
        n_by_ell: Dict[int, List[int]] = {}
        for (ell, n_idx) in f_on_grid.keys():
            n_by_ell.setdefault(ell, []).append(n_idx)
        tasks_lm = []
        for ell in sorted(n_by_ell):
            for m_val in range(-ell, ell + 1):
                if (ell, m_val) in a_lm:
                    tasks_lm.append((ell, m_val, sorted(n_by_ell[ell]), a_lm[(ell, m_val)]))
        n_triples = sum(len(t[2]) for t in tasks_lm)
        # Use pool initializer to populate global variables in each worker.
        # Without an initializer the ``spawn`` start method (default on Windows) would
        # create child processes with empty module state, causing KeyError.
        with _mp_context().Pool(
            processes=num_workers,
            initializer=_init_cnlm_worker_lm,
            initargs=(r_int, weight, dr_int, n_times, r_al_arr, r_interp_kind, f_on_grid),
        ) as pool:
            total_jobs = n_triples
            done = 0
            last_percent_print = -1
            worker_counts: Dict[str, int] = {}
            worker_alias: Dict[str, str] = {}
            alias_order: List[str] = []
            # Single-line progress renderer for parallel mode
            def _render_progress_parallel(percent: int, done_local: int) -> None:
                bar_len = 30
                filled = int(bar_len * max(0, min(100, percent)) / 100)
                bar = '=' * filled + '.' * (bar_len - filled)
                parts: List[str] = []
                for wn in alias_order:
                    al = worker_alias[wn]
                    parts.append(f"{al}:{worker_counts.get(al, 0)}")
                per_worker = ' '.join(parts)
                msg = f"[{bar}] {percent:3d}% ({done_local}/{total_jobs}) | {per_worker}"
                try:
                    from IPython.display import clear_output  # type: ignore
                    clear_output(wait=True)
                    print(msg, end='', flush=True)
                except Exception:
                    sys.stdout.write('\r' + msg)
                    sys.stdout.flush()
            for results_lm, wname in pool.imap_unordered(_cnlm_worker_lm, tasks_lm):
                for key, c_arr in results_lm:
                    c_dict[key] = c_arr
                if progress:
                    done += len(results_lm)
                    # Map raw worker name to short alias W1, W2, ... for readability
                    if wname not in worker_alias:
                        alias = f"W{len(worker_alias) + 1}"
                        worker_alias[wname] = alias
                        alias_order.append(wname)
                    alias = worker_alias[wname]
                    worker_counts[alias] = worker_counts.get(alias, 0) + len(results_lm)
                    percent = int((done * 100) / total_jobs) if total_jobs > 0 else 100
                    if percent != last_percent_print:
                        _render_progress_parallel(percent, done)
                        last_percent_print = percent
            if progress and total_jobs > 0:
                # Ensure the bar is full and move to next line
                _render_progress_parallel(100, done)
                print()
        # Do not attempt progress output in parallel mode
    # After computing c_dict, perform time resampling if requested
    if need_time_resample:
        # Use the unified resampler that supports pchip/akima/natural cubic, etc.
        c_dict = resample_cnlm_time(c_dict, t_al, t_ret, method=kind_time)
    return t_ret, c_dict


def resample_cnlm_time(
    c_dict: Dict[Tuple[int, int, int], np.ndarray],
    t_in: np.ndarray,
    t_out: np.ndarray,
    method: str = 'pchip',
) -> Dict[Tuple[int, int, int], np.ndarray]:
    r"""Resample precomputed ``c_{nlm}(t)`` onto a new time grid.

    Supported methods:

    - 'nearest', 'linear', 'cubic' (scipy.interpolate.interp1d).  Since
      version 0.3.0 'nearest' is pure nearest-neighbour sampling; before, it
      silently applied the same Savitzky–Golay smoothing as
      'nearest-savgol'.  Use 'nearest-savgol' (as the paper notebooks do) to
      reproduce the historical dense files.
    - 'pchip'  (shape-preserving cubic Hermite; smooth, no ringing; stable at endpoints)
    - 'akima'  (Akima1DInterpolator; robust to noise, reduces oscillations)
    - 'cspline-natural' (CubicSpline with natural boundary conditions)
    - 'nearest-smooth' / 'nearest-savgol' (pipeline: nearest then Savitzky–Golay
      smoothing of |c|; the phase stays piecewise constant, so the result is
      not suitable for phase or spectral work)

    Will gracefully degrade order when sample count is insufficient.

    Parameters
    ----------
    c_dict : dict
        Mapping from ``(n, \ell, m)`` to complex arrays sampled on ``t_in``.

    t_in : np.ndarray
        The original time grid corresponding to the input series. Must be
        one‑dimensional and nondecreasing.

    t_out : np.ndarray
        The desired output time grid. Must be one‑dimensional and
        nondecreasing.

    method : str, optional
        One of {'nearest','linear','cubic','pchip','akima','cspline-natural'}.

    Returns
    -------
    dict
        A new dictionary on the same keys with arrays sampled on ``t_out``.
    """
    t_in_arr = np.asarray(t_in, dtype=np.float64)
    t_out_arr = np.asarray(t_out, dtype=np.float64)
    if t_in_arr.ndim != 1 or not np.all(np.diff(t_in_arr) >= 0):
        raise ValueError("t_in must be one‑dimensional and nondecreasing.")
    if t_out_arr.ndim != 1 or not np.all(np.diff(t_out_arr) >= 0):
        raise ValueError("t_out must be one‑dimensional and nondecreasing.")
    n_in = len(t_in_arr)

    def _eval_one(y: np.ndarray) -> np.ndarray:
        # Complex handling: interpolate real and imaginary parts separately
        if np.iscomplexobj(y):
            # For nearest-smooth pipeline we must treat complex as a whole to preserve phase.
            m_lower = (method or 'linear').lower()
            if m_lower in ('nearest-smooth', 'nearest-savgol'):
                # 1) nearest on complex series
                f_nn = _interp.interp1d(
                    t_in_arr, y, kind='nearest', bounds_error=False,
                    fill_value=(y[0], y[-1]), assume_sorted=True
                )
                y_nn = f_nn(t_out_arr)
                # 2) smooth magnitude |c| with Savitzky–Golay
                try:
                    from scipy.signal import savgol_filter  # type: ignore
                except Exception:
                    savgol_filter = None  # type: ignore
                amp = np.abs(y_nn)
                # Heuristic window: ~ 2*densification_factor + 1, min 5, odd
                dens = (len(t_out_arr) - 1) / max(1, (len(t_in_arr) - 1))
                win = int(max(5, 2 * int(max(1, round(dens))) + 1))
                # Ensure window <= len(t_out_arr) and odd
                if win >= len(t_out_arr):
                    win = len(t_out_arr) - (1 - len(t_out_arr) % 2)
                    if win < 5:
                        win = 5 if len(t_out_arr) >= 5 else (len(t_out_arr) | 1)
                if win % 2 == 0:
                    win = max(5, win + 1)
                poly = 2 if win > 5 else 1
                if savgol_filter is not None and win >= 3 and len(t_out_arr) >= win:
                    amp_s = savgol_filter(amp, window_length=win, polyorder=poly, mode='interp')
                else:
                    # Fallback: simple moving average with reflect padding
                    k = win
                    pad = k // 2
                    xpad = np.pad(amp, pad_width=pad, mode='reflect')
                    kernel = np.ones(k, dtype=np.float64) / k
                    amp_s = np.convolve(xpad, kernel, mode='valid')
                # 3) reconstruct with preserved phase
                eps = 1e-12
                scale = amp_s / np.maximum(amp, eps)
                return y_nn * scale
            # For other methods, split real/imag
            yr = _eval_one(np.real(y))
            yi = _eval_one(np.imag(y))
            return yr + 1j * yi

        m = (method or 'linear').lower()
        eff = m
        # Degrade order based on available samples
        if m in ('cubic',):
            if n_in < 4:
                eff = 'linear'
        elif m in ('pchip',):
            if n_in < 2:
                return np.full_like(t_out_arr, y[0], dtype=y.dtype)
        elif m in ('akima',):
            # Akima performs best with >=5 points
            if n_in < 5:
                eff = 'pchip' if n_in >= 2 else 'linear'
        elif m in ('cspline-natural',):
            if n_in < 3:
                eff = 'linear'
        elif m in ('nearest-smooth', 'nearest-savgol'):
            # Complex branch handled above; for real, run pipeline here
            f_nn = _interp.interp1d(
                t_in_arr, y, kind='nearest', bounds_error=False,
                fill_value=(y[0], y[-1]), assume_sorted=True
            )
            y_nn = f_nn(t_out_arr)
            try:
                from scipy.signal import savgol_filter  # type: ignore
            except Exception:
                savgol_filter = None  # type: ignore
            dens = (len(t_out_arr) - 1) / max(1, (len(t_in_arr) - 1))
            win = int(max(5, 2 * int(max(1, round(dens))) + 1))
            if win >= len(t_out_arr):
                win = len(t_out_arr) - (1 - len(t_out_arr) % 2)
                if win < 5:
                    win = 5 if len(t_out_arr) >= 5 else (len(t_out_arr) | 1)
            if win % 2 == 0:
                win = max(5, win + 1)
            poly = 2 if win > 5 else 1
            if savgol_filter is not None and win >= 3 and len(t_out_arr) >= win:
                return savgol_filter(y_nn, window_length=win, polyorder=poly, mode='interp')
            # Fallback moving average
            k = win
            pad = k // 2
            xpad = np.pad(y_nn, pad_width=pad, mode='reflect')
            kernel = np.ones(k, dtype=np.float64) / k
            return np.convolve(xpad, kernel, mode='valid')
        elif m in ('linear', 'nearest'):
            pass
        else:
            raise ValueError("method must be one of 'nearest','linear','cubic','pchip','akima','cspline-natural',"
                             "'nearest-smooth','nearest-savgol'.")

        if eff in ('nearest', 'linear', 'cubic'):
            f = _interp.interp1d(
                t_in_arr, y, kind=eff, bounds_error=False,
                fill_value=(y[0], y[-1]), assume_sorted=True
            )
            return f(t_out_arr)
        if eff == 'pchip':
            f = _interp.PchipInterpolator(t_in_arr, y, extrapolate=False)
            out = f(t_out_arr)
            # Fill possible NaNs (outside range) with end values
            if np.any(~np.isfinite(out)):
                out = np.where(
                    ~np.isfinite(out),
                    np.interp(t_out_arr, [t_in_arr[0], t_in_arr[-1]], [y[0], y[-1]]),
                    out,
                )
            return out
        if eff == 'akima':
            f = _interp.Akima1DInterpolator(t_in_arr, y)
            out = f(t_out_arr)
            return out
        if eff == 'cspline-natural':
            f = _interp.CubicSpline(t_in_arr, y, bc_type='natural', extrapolate=False)
            out = f(t_out_arr)
            if np.any(~np.isfinite(out)):
                out = np.where(
                    ~np.isfinite(out),
                    np.interp(t_out_arr, [t_in_arr[0], t_in_arr[-1]], [y[0], y[-1]]),
                    out,
                )
            return out
        # Fallback to linear (should not reach here)
        f = _interp.interp1d(
            t_in_arr, y, kind='linear', bounds_error=False,
            fill_value=(y[0], y[-1]), assume_sorted=True
        )
        return f(t_out_arr)

    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    for key, series in c_dict.items():
        out[key] = _eval_one(np.asarray(series))
    return out


def save_cnlm_to_h5(
    c_dict: Dict[Tuple[int, int, int], np.ndarray],
    t_array: np.ndarray,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save computed ``c_{nlm}(t)`` coefficients to an HDF5 file.

    The output hierarchy is organised primarily by radial quantum
    number ``n`` followed by angular momentum ``ℓ``.  Under the root
    there is a group ``/n``.  Within ``/n`` each radial quantum
    number ``n`` has a subgroup.  Under each ``n`` subgroup there is
    a group ``l``.  Inside ``/n/<n>/l/<ℓ>`` the datasets ``m_<m>``
    store the complex time series of length ``len(t_array)``.  A
    top‑level dataset ``t`` stores the time grid.  Any metadata
    provided will be attached as attributes on the root group.  This
    layout mirrors the key ordering ``(n, ℓ, m)`` used in ``c_dict``.

    Parameters
    ----------
    c_dict : dict
        Mapping from ``(n, ℓ, m)`` tuples (the key order returned by
        :func:`compute_cnlm`) to one‑dimensional complex arrays of the same
        length as ``t_array``.

    t_array : np.ndarray
        One‑dimensional array of time values.  Must match the length of
        the coefficient arrays.

    output_path : str
        Path of the HDF5 file to create.  If the file exists it will be
        overwritten.

    metadata : dict, optional
        Additional metadata to attach as attributes on the root group.
        Values that HDF5 cannot store natively (dicts, mixed lists, None)
        are stored as JSON strings.  Attributes never change the group and
        dataset layout.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.

    ValueError
        If the time arrays associated with the coefficients are not all
        the same length as ``t_array``.
    """
    if h5py is None:
        raise ImportError(
            "h5py is required to save c_{nlm} data but is not installed."
        )
    t_arr = np.asarray(t_array, dtype=np.float64)
    if t_arr.ndim != 1:
        raise ValueError("t_array must be one‑dimensional.")
    # Determine number of time points
    n_t = len(t_arr)
    # Validate lengths
    for key, arr in c_dict.items():
        if arr.ndim != 1 or len(arr) != n_t:
            raise ValueError(
                f"Coefficient for key {key} has length {len(arr)}, expected {n_t}."
            )
    # Create file and write data
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('t', data=t_arr, dtype='float64')
        # Write metadata
        if metadata is not None:
            from ._h5meta import write_attrs
            write_attrs(f.attrs, metadata)
        # The hierarchical structure: /n/<n>/l/<ℓ>/m_<m>
        for (n_val, ell_val, m_val), arr in c_dict.items():
            # Create group path '/n/<n_val>/l/<ell_val>'
            grp_path = f'n/{n_val}/l/{ell_val}'
            # Ensure parent groups exist
            if grp_path not in f:
                f.create_group(grp_path)
            grp = f[grp_path]
            ds_name = f'm_{m_val}'
            grp.create_dataset(ds_name, data=arr, dtype='complex128')


def load_cnlm_from_h5(
    path: str,
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], np.ndarray], Dict[str, Any]]:
    """Load previously saved ``c_{nlm}(t)`` data from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to the HDF5 file produced by ``save_cnlm_to_h5``.

    Returns
    -------
    t : np.ndarray
        One‑dimensional array containing the time grid.

    c_dict : dict
        Mapping from ``(n, ℓ, m)`` to one‑dimensional complex arrays of
        shape ``(len(t),)`` containing the coefficients.

    attrs : dict
        Dictionary of attributes stored in the root of the HDF5 file.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.

    KeyError
        If the file lacks the expected groups or datasets.
    """
    if h5py is None:
        raise ImportError(
            "h5py is required to load c_{nlm} data but is not installed."
        )
    c_dict: Dict[Tuple[int, int, int], np.ndarray] = {}
    attrs: Dict[str, Any] = {}
    with h5py.File(path, 'r') as f:
        # Retrieve time array
        if 't' not in f:
            raise KeyError(f"Dataset 't' not found in {path}.")
        t_arr = np.asarray(f['t'], dtype=np.float64)
        # Load attributes
        for k, v in f.attrs.items():
            attrs[k] = v
        # Expect group 'n' at the root
        if 'n' not in f:
            raise KeyError(f"Group 'n' not found in {path}.")
        n_grp_root = f['n']
        for n_key in n_grp_root.keys():
            # n_key should be a string representing the radial quantum number
            n_val = int(n_key)
            l_container = n_grp_root[n_key]
            if 'l' not in l_container:
                raise KeyError(f"Group 'l' not found under n/{n_key} in {path}.")
            l_grp = l_container['l']
            for l_key in l_grp.keys():
                ell_val = int(l_key)
                m_grp = l_grp[l_key]
                for m_ds_name in m_grp.keys():
                    if not m_ds_name.startswith('m_'):
                        continue
                    m_val = int(m_ds_name.split('_')[1])
                    data = np.asarray(m_grp[m_ds_name], dtype=np.complex128)
                    if data.ndim != 1 or data.shape[0] != t_arr.shape[0]:
                        raise ValueError(
                            f"Dataset '{m_ds_name}' has inconsistent shape {data.shape}."
                        )
                    c_dict[(n_val, ell_val, m_val)] = data
    return t_arr, c_dict, attrs