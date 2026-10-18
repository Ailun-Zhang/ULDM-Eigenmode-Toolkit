r"""One-pass spherical-harmonic decomposition a_lm(r) of one wavefunction cube.

This module is a fast, dependency-light re-implementation of the historical
per-(snapshot, shell) kernel ``functions3.process_file`` that produced every
published a_lm / c_nlm file.  For one snapshot cube it

1. computes the |psi|^2-weighted centre of mass once (same formula and the
   same floating-point operations as ``functions3.calculate_mass_center``),
2. computes the cubic B-spline coefficients of Re(psi) and Im(psi) once
   (``scipy.ndimage.spline_filter(order=3, mode='constant')``, exactly the
   call ``map_coordinates(prefilter=True)`` makes internally),
3. samples every shell with ``map_coordinates(order=3, mode='constant',
   cval=0, prefilter=False)`` on the same N x 2N (theta, phi) grid, using a
   thread pool over shells and point chunks (``map_coordinates`` releases the
   GIL; each output point is computed independently, so chunking does not
   change any value), and
4. calls ``pyshtools.expand.SHExpandDHC(grid, norm=4, sampling=2,
   csphase=-1, lmax_calc=l_max)`` on each shell (serially, in the calling
   thread).

With ``angular_grid='legacy'`` and ``center='com'`` the coefficients are
**bitwise identical** to what ``process_file`` writes into its JSON files,
for double (complex128/float64) and for single-precision (complex64/float32)
cubes: like ``map_coordinates(wf.real, ...)`` in ``process_file``, the shell
samples are returned in the real dtype of the cube
(``tests/test_alm_kernel.py`` checks this, and the equivalence script in the
WP2 report checks the full a_lm -> c_nlm chain).  The historical kernel
reloaded the cube, rebuilt three N^3 meshgrids and re-ran the spline
prefilter twice for every shell; here this is done once per snapshot.

Coordinate convention (identical to ``functions3``)
---------------------------------------------------
Voxel ``(i, j, k)`` of the cube sits at ``(i*dx, j*dx, k*dx)``.  The
expansion centre ``c`` is given in the same units (``center='com'`` uses the
density-weighted mean).  Shell point ``(r, theta, phi)`` is sampled at index
coordinates ``((r sin(theta) cos(phi) + c_x)/dx, ...)``.  ``r_values`` and
``dx`` must use the same length unit (kpc in the historical scripts); the
unit itself never enters the coefficients.

Output layout
-------------
``compute_alm_cube`` returns a complex128 array of shape
``(n_r, (l_max+1)**2)`` whose columns are ordered ``l = 0..l_max`` and, within
each ``l``, ``m = -l..l`` (the layout of ``alm_utils.reorganize_coeffs`` and of
the ``ylm`` datasets in the a_lm HDF5 file, before the multiplication by
``counts``).  ``return_raw=True`` returns the raw ``(n_r, 2, l_max+1, l_max+1)``
SHExpandDHC arrays instead (index 0 holds m >= 0, index 1 holds m < 0 at
``[1, l, |m|]``).  Normalisation: orthonormal complex Y_lm with the
Condon-Shortley phase, i.e. ``a_lm(r) = int psi(r, Omega) Y_lm^*(Omega) dOmega``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as _ndimage

from ._compat import default_workers

__all__ = [
    "ANGULAR_GRIDS",
    "angular_grid_1d",
    "lm_list",
    "flatten_shtools_coeffs",
    "mass_center_index_space",
    "prefilter_cube",
    "AlmKernel",
    "compute_alm_cube",
]

ANGULAR_GRIDS = ("legacy", "dh")


def angular_grid_1d(num_points: int, angular_grid: str = "legacy") -> Tuple[np.ndarray, np.ndarray]:
    """1-D (theta, phi) samples of the N x 2N sphere grid passed to SHExpandDHC.

    ``'legacy'`` (default, used for all published data): ``theta =
    linspace(0, pi, N)`` and ``phi = linspace(0, 2 pi, 2N)``, both endpoints
    included.  This is *not* the Driscoll-Healy grid that
    ``SHExpandDHC(sampling=2)`` assumes; it slightly stretches the sphere
    (see the WP2 report for the measured effect).  ``'dh'``: ``theta_i =
    pi i/N``, ``phi_j = pi j/N`` (the correct Driscoll-Healy grid).
    """
    N = int(num_points)
    if angular_grid == "legacy":
        return np.linspace(0, np.pi, N), np.linspace(0, 2 * np.pi, 2 * N)
    if angular_grid == "dh":
        return np.arange(N) * np.pi / N, np.arange(2 * N) * np.pi / N
    raise ValueError(f"angular_grid must be one of {ANGULAR_GRIDS}, got {angular_grid!r}")


def lm_list(l_max: int) -> List[Tuple[int, int]]:
    """(l, m) pairs in column order: l = 0..l_max, m = -l..l."""
    return [(l, m) for l in range(int(l_max) + 1) for m in range(-l, l + 1)]


def _flat_index_arrays(l_max: int):
    i0, il, im = [], [], []
    for l, m in lm_list(l_max):
        if m >= 0:
            i0.append(0); il.append(l); im.append(m)
        else:
            i0.append(1); il.append(l); im.append(-m)
    return np.array(i0), np.array(il), np.array(im)


def flatten_shtools_coeffs(raw: np.ndarray, l_max: int) -> np.ndarray:
    """Reorder SHExpandDHC output ``(..., 2, L+1, L+1)`` to ``(..., (L+1)**2)``.

    Pure copy of values (no arithmetic), identical to
    ``alm_utils.reorganize_coeffs`` applied to each leading index.
    """
    raw = np.asarray(raw)
    i0, il, im = _flat_index_arrays(l_max)
    return np.ascontiguousarray(raw[..., i0, il, im])


def mass_center_index_space(psi_cube: np.ndarray, dx: float) -> np.ndarray:
    """|psi|^2-weighted centre of mass, voxel (i,j,k) at (i dx, j dx, k dx).

    Performs exactly the floating-point operations of
    ``functions3.calculate_mass_center`` (so the result is bitwise equal) but
    broadcasts a 1-D coordinate vector instead of materialising three N^3
    meshgrids (one N^3 temporary at a time instead of three).
    """
    density_data = np.abs(psi_cube) ** 2
    n = density_data.shape[0]
    x = np.arange(n) * dx
    total_mass = np.sum(density_data)
    cx = np.sum(x[:, None, None] * density_data) / total_mass
    cy = np.sum(x[None, :, None] * density_data) / total_mass
    cz = np.sum(x[None, None, :] * density_data) / total_mass
    return np.array([cx, cy, cz])


def prefilter_cube(psi_cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic B-spline coefficients of Re(psi) and Im(psi) for mode='constant'.

    This is the call ``scipy.ndimage.map_coordinates(..., order=3,
    mode='constant', prefilter=True)`` performs internally on every call
    (for mode='constant' no pre-padding is applied), so sampling the returned
    arrays with ``prefilter=False`` gives bitwise identical values.
    """
    psi_cube = np.asarray(psi_cube)
    re = _ndimage.spline_filter(psi_cube.real, 3, output=np.float64, mode="constant")
    if np.iscomplexobj(psi_cube):
        im = _ndimage.spline_filter(psi_cube.imag, 3, output=np.float64, mode="constant")
    else:
        im = np.zeros_like(re)
    return re, im


class AlmKernel:
    """Reusable one-pass a_lm(r) kernel (fixed shells and angular grid).

    Parameters
    ----------
    r_values : sequence of float
        Shell radii, same length unit as the ``dx`` passed to :meth:`compute`.
        The historical scripts used ``linspace(0, r_max, n_r)`` with
        ``r[0] = 1e-5`` (kpc).
    l_max : int
        Maximum degree passed to ``SHExpandDHC(lmax_calc=l_max)``.
    num_points : int
        N of the N x 2N angular grid (even, ``l_max <= N/2 - 1``).  Paper: 300.
    angular_grid : {'legacy', 'dh'}
        Sphere sampling, see :func:`angular_grid_1d`.  Default 'legacy'.
    workers : int, optional
        Threads for the interpolation step.  ``None`` uses
        :func:`Eig_ULDM_packages._compat.default_workers` (environment
        variables ``ULDM_NUM_WORKERS``/``PBS_NCPUS``/``SLURM_CPUS_PER_TASK``/
        ``OMP_NUM_THREADS``, else the CPU affinity); ``1`` runs serially.
    chunk_points : int, optional
        Maximum number of sphere points per interpolation task.  By default a
        shell is split only when there are fewer shells than threads.  Values
        never depend on the chunking.
    """

    def __init__(
        self,
        r_values: Sequence[float],
        l_max: int,
        num_points: int = 300,
        angular_grid: str = "legacy",
        workers: Optional[int] = None,
        chunk_points: Optional[int] = None,
    ) -> None:
        self.r_values = np.asarray(r_values, dtype=np.float64).ravel()
        if self.r_values.size == 0:
            raise ValueError("r_values must contain at least one radius.")
        self.l_max = int(l_max)
        self.num_points = int(num_points)
        if self.num_points < 2 or self.num_points % 2:
            raise ValueError("num_points must be an even integer >= 2 (SHExpandDHC N x 2N grid).")
        if self.l_max < 0 or self.l_max > self.num_points // 2 - 1:
            raise ValueError(f"l_max must satisfy 0 <= l_max <= num_points/2 - 1 = {self.num_points // 2 - 1}.")
        self.angular_grid = str(angular_grid)
        theta, phi = angular_grid_1d(self.num_points, self.angular_grid)
        # Same (N, 2N) 'ij' meshgrid as functions3.process_file; the
        # trigonometric factors are evaluated once on these identical arrays.
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        self.grid_shape = theta_grid.shape
        self._sin_t = np.sin(theta_grid).ravel()
        self._cos_t = np.cos(theta_grid).ravel()
        self._cos_p = np.cos(phi_grid).ravel()
        self._sin_p = np.sin(phi_grid).ravel()
        self.n_points = self._sin_t.size
        self.workers = default_workers(workers)
        self.chunk_points = None if chunk_points is None else max(1, int(chunk_points))

    # ------------------------------------------------------------------
    def _chunks(self, n_shell: Optional[int] = None) -> List[Tuple[int, int]]:
        n_shell = self.r_values.size if n_shell is None else int(n_shell)
        if self.chunk_points is not None:
            size = self.chunk_points
        elif self.workers > n_shell:
            n_split = -(-self.workers // n_shell)
            size = -(-self.n_points // n_split)
        else:
            size = self.n_points
        return [(a, min(a + size, self.n_points)) for a in range(0, self.n_points, size)]

    def _sample(self, coef_re, coef_im, center, dx, r, a, b, out_dtype=None):
        # Operation order mirrors functions3.compute_wave_function_on_sphere_mapcoord:
        #   x = r * sin(theta) * cos(phi); y = r * sin(theta) * sin(phi); z = r * cos(theta)
        #   xi = (x + c_x) / dx, ...
        x = r * self._sin_t[a:b] * self._cos_p[a:b]
        y = r * self._sin_t[a:b] * self._sin_p[a:b]
        z = r * self._cos_t[a:b]
        xi = (x + center[0]) / dx
        yi = (y + center[1]) / dx
        zi = (z + center[2]) / dx
        coords = np.vstack([xi, yi, zi])
        # ``output=out_dtype``: process_file calls map_coordinates(wf.real, ...), whose output dtype follows
        # the cube (float32 samples for a complex64 cube; the spline arithmetic is float64 either way).
        real_vals = _ndimage.map_coordinates(coef_re, coords, output=out_dtype, order=3, mode="constant", cval=0.0,
                                             prefilter=False)
        imag_vals = _ndimage.map_coordinates(coef_im, coords, output=out_dtype, order=3, mode="constant", cval=0.0,
                                             prefilter=False)
        return real_vals, imag_vals

    def sample_shells(
        self,
        psi_cube: np.ndarray,
        dx: float,
        center: Union[str, Iterable[float]] = "com",
        prefiltered: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        shells: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate psi on every shell (or on the shell indices ``shells``).

        Returns ``(values, center)`` where ``values`` has shape
        ``(n_shells, N, 2N)`` complex128 and ``center`` is the expansion centre used.
        """
        psi_cube = np.asarray(psi_cube)
        if psi_cube.ndim != 3 or len(set(psi_cube.shape)) != 1:
            raise ValueError(f"psi_cube must be a cubic 3-D array, got shape {psi_cube.shape}.")
        dx = float(dx)
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError("dx must be a positive finite number.")
        if isinstance(center, str):
            if center != "com":
                raise ValueError("center must be 'com' or an explicit 3-vector.")
            c = mass_center_index_space(psi_cube, dx)
        else:
            c = np.asarray(list(center), dtype=np.float64)
            if c.shape != (3,):
                raise ValueError("an explicit center must have 3 components.")
        if not np.all(np.isfinite(c)):
            raise ValueError(f"expansion centre is not finite: {c} (empty or invalid cube?)")
        coef_re, coef_im = prefilter_cube(psi_cube) if prefiltered is None else prefiltered
        # sample dtype of process_file: the real dtype of the cube (float64 unless the cube is single precision)
        real_dtype = psi_cube.real.dtype
        out_dtype = real_dtype if real_dtype in (np.float32, np.float64) else np.float64

        shell_idx = list(range(self.r_values.size)) if shells is None else [int(i) for i in shells]
        chunks = self._chunks(len(shell_idx))
        tasks = [(j, i, a, b) for j, i in enumerate(shell_idx) for (a, b) in chunks]
        out = np.empty((len(shell_idx), self.n_points), dtype=np.complex128)

        def run(task):
            j, i, a, b = task
            re_v, im_v = self._sample(coef_re, coef_im, c, dx, self.r_values[i], a, b, out_dtype)
            # Same complex assembly as functions3: real_vals + 1j*imag_vals
            out[j, a:b] = re_v + 1j * im_v

        if self.workers <= 1 or len(tasks) == 1:
            for t in tasks:
                run(t)
        else:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(tasks))) as ex:
                for _ in ex.map(run, tasks):
                    pass
        return out.reshape((len(shell_idx),) + self.grid_shape), c

    def compute(
        self,
        psi_cube: np.ndarray,
        dx: float,
        center: Union[str, Iterable[float]] = "com",
        return_raw: bool = False,
        prefiltered: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        return_center: bool = False,
        shell_block: Optional[int] = None,
    ):
        """a_lm for all shells of one snapshot cube.

        Returns an array ``(n_r, (l_max+1)**2)`` complex128 (or the raw
        ``(n_r, 2, l_max+1, l_max+1)`` SHExpandDHC output with
        ``return_raw=True``); with ``return_center=True`` a tuple
        ``(coefficients, center)``.

        ``shell_block`` (optional) samples and expands the shells in blocks of
        this many shells, so only ``shell_block x N x 2N`` sphere values are held
        at a time instead of ``n_r x N x 2N`` (740 MB for the paper's 257 shells
        at N = 300).  Every value is computed exactly as without blocks (each
        shell is independent), so the result is bitwise identical.
        """
        import pyshtools as pysh  # lazy: only needed when coefficients are computed

        n_r = self.r_values.size
        raw = np.empty((n_r, 2, self.l_max + 1, self.l_max + 1), dtype=np.complex128)
        if shell_block is None or int(shell_block) >= n_r:
            values, c = self.sample_shells(psi_cube, dx, center=center, prefiltered=prefiltered)
            for i in range(n_r):
                raw[i] = pysh.expand.SHExpandDHC(values[i], norm=4, sampling=2, csphase=-1, lmax_calc=self.l_max)
        else:
            block = max(1, int(shell_block))
            psi_cube = np.asarray(psi_cube)
            if isinstance(center, str):
                if center != "com":
                    raise ValueError("center must be 'com' or an explicit 3-vector.")
                if psi_cube.ndim != 3 or len(set(psi_cube.shape)) != 1:
                    raise ValueError(f"psi_cube must be a cubic 3-D array, got shape {psi_cube.shape}.")
                c = mass_center_index_space(psi_cube, float(dx))
            else:
                c = np.asarray(list(center), dtype=np.float64)
            pre = prefilter_cube(psi_cube) if prefiltered is None else prefiltered
            for start in range(0, n_r, block):
                idx = range(start, min(start + block, n_r))
                values, c = self.sample_shells(psi_cube, dx, center=c, prefiltered=pre, shells=idx)
                for j, i in enumerate(idx):
                    raw[i] = pysh.expand.SHExpandDHC(values[j], norm=4, sampling=2, csphase=-1,
                                                     lmax_calc=self.l_max)
                del values
        res = raw if return_raw else flatten_shtools_coeffs(raw, self.l_max)
        return (res, c) if return_center else res


def compute_alm_cube(
    psi_cube: np.ndarray,
    dx: float,
    r_values: Sequence[float],
    l_max: int,
    num_points: int = 300,
    angular_grid: str = "legacy",
    center: Union[str, Iterable[float]] = "com",
    workers: Optional[int] = None,
    return_raw: bool = False,
    chunk_points: Optional[int] = None,
) -> np.ndarray:
    """a_lm(r) of one cube for all shells in one pass (see module docstring).

    Parameters
    ----------
    psi_cube : (N, N, N) complex array
        Wavefunction cube, e.g. a PyUL ``Outputs/3WfnRS/P3R_#NNN.npy`` file.
    dx : float
        Grid spacing of the cube in the length unit of ``r_values``.  For
        3WfnRS this is the *resampled* box side divided by ``resolR``.
    r_values : sequence of float
        Shell radii.
    l_max, num_points, angular_grid, workers, chunk_points :
        See :class:`AlmKernel`.  Defaults reproduce the paper (300, 'legacy').
    center : 'com' or 3-vector
        'com' (default, as ``functions3.process_file``) or an explicit
        centre in the same units as ``dx`` with voxel ``i`` at ``i*dx``.
    return_raw : bool
        Return raw SHExpandDHC arrays ``(n_r, 2, l_max+1, l_max+1)``.

    Returns
    -------
    np.ndarray
        ``(n_r, (l_max+1)**2)`` complex128, columns l = 0..l_max, m = -l..l.
    """
    kernel = AlmKernel(r_values, l_max, num_points=num_points, angular_grid=angular_grid,
                       workers=workers, chunk_points=chunk_points)
    return kernel.compute(psi_cube, dx, center=center, return_raw=return_raw)
