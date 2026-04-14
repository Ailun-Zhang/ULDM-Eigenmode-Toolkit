"""
Python implementation of the ultralight‑dark‑matter (ULDM) radial eigenvalue
solver.

This module provides a set of functions that collectively reproduce the
behaviour of the reference Mathematica/MATLAB pipeline (``ULDMEig.m``)
for solving the radial Schrödinger equation of self‑gravitating scalar
solitons.  The implementation is deliberately conservative: it uses
identical finite‑difference stencils, boundary conditions and scaling
conventions to those found in the original code, and it eschews
optimisations that might change the resulting eigenvalues or
eigenfunctions.  Extensive commentary accompanies each function to
explain the underlying physics, numerical choices, and potential edge
cases.

Users who wish to experiment with alternative discretisations or
improvements (e.g., higher‑order stencils, adaptive grids or FFT‑based
Poisson solves) should do so only after verifying that the baseline
version exactly reproduces the reference outputs.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Tuple, Any, Optional, List

import numpy as np
import scipy.interpolate as _interp
import scipy.integrate as _integrate
import scipy.sparse as _sparse
import scipy.sparse.linalg as _sparse_linalg
try:
    import h5py  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - h5py might not be installed in restricted environments
    # In many educational environments the HDF5 Python bindings are not
    # installed by default.  We still allow the module to be imported but
    # raise a more informative error when HDF5 interaction is required.
    h5py = None  # type: ignore[assignment]


def load_soliton_file(h5_path: str) -> Dict[str, Any]:
    """Load a dimensionless soliton profile from an HDF5 file.

    The reference ULDM pipeline stores the ground‑state soliton profile
    in a very simple HDF5 container under the key ``/profile``.  Two
    scalar attributes accompany the dataset: ``dr``, the uniform step
    size of the tabulated grid, and ``mass0``, the mass (in dimensionless
    units) associated with the tabulated profile.  The Mathematica code
    uses these values to reconstruct a continuous interpolation ``ψ(y)``
    on ``y ∈ [0, y_max]`` where ``y = i * dr`` for ``i = 0..n-1``.

    This function reproduces that behaviour.  It reads the raw profile,
    constructs a radial array ``y`` of the same length using the stored
    ``dr``, and returns a dictionary with the maximum radial extent
    ``ymax``, a one–dimensional interpolation ``psi`` and a callable
    ``alpha_func`` that maps a soliton mass ``m`` to its scaling
    parameter ``α = (m / mass0)^2``.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file containing the soliton data.  The file
        must contain a dataset ``/profile`` and attributes ``dr`` and
        ``mass0``.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``'ymax'`` (float): the maximum value of the tabulated radial
          coordinate ``y``.
        - ``'psi'`` (callable): a one–dimensional cubic spline
          interpolant ``psi(y)`` that returns the soliton amplitude at
          arbitrary ``y`` (values outside ``[0,ymax]`` return zero).
        - ``'alpha_func'`` (callable): a function of one argument
          ``m`` returning ``(m / mass0)^2``, used for scaling the
          profile to arbitrary soliton masses.

    Raises
    ------
    ImportError
        If ``h5py`` is unavailable when attempting to read the file.

    KeyError
        If ``/profile``, or either attribute ``dr`` or ``mass0``, is
        missing from the file.

    ValueError
        If ``dr`` or ``mass0`` cannot be cast to positive floats.
    """
    if h5py is None:
        raise ImportError(
            "h5py is required to load the soliton profile but is not installed."
        )
    with h5py.File(h5_path, "r") as f:
        if "profile" not in f:
            raise KeyError(
                f"Dataset '/profile' not found in {h5_path}."
            )
        prof = np.asarray(f["profile"], dtype=np.float64)
        try:
            dr = float(f["profile"].attrs["dr"])
            mass0 = float(f["profile"].attrs["mass0"])
        except KeyError as e:
            raise KeyError(
                f"Required attribute {e} missing from '/profile' in {h5_path}."
            )
        except Exception:
            raise ValueError(
                "Attributes 'dr' and 'mass0' must be convertible to floats."
            )
    if dr <= 0 or mass0 <= 0:
        raise ValueError("Attributes 'dr' and 'mass0' must be positive.")
    n = len(prof)
    y = np.arange(n, dtype=np.float64) * dr
    ymax = y[-1]
    # Construct an interpolation of ψ(y).  Outside the domain we return 0.
    # Use a cubic spline with no extrapolation.  The 'extrapolate=False'
    # option ensures values beyond [0,ymax] yield NaN; we catch these
    # later and set to zero.
    spline = _interp.CubicSpline(y, prof, bc_type="natural", extrapolate=False)
    def psi_func(x: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        # Evaluate the spline; replace NaN outside the domain with 0.
        vals = spline(x_arr)
        # Where extrapolated values are NaN, set to zero.
        mask = np.isnan(vals)
        if np.any(mask):
            vals = vals.copy()
            vals[mask] = 0.0
        return vals
    # Scaling function α(m) = (m/mass0)^2
    def alpha_func(m: float) -> float:
        m = float(m)
        if m <= 0:
            raise ValueError("Mass must be positive in alpha_func.")
        return (m / mass0) ** 2
    return {
        "ymax": ymax,
        "psi": psi_func,
        "alpha_func": alpha_func,
    }


def soliton_density_function(m: float, sol: Dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Construct the physical density ``ρ(r)`` for a soliton of mass ``m``.

    The loaded soliton profile ``sol`` supplies a dimensionless ground
    state ``ψ(y)`` defined on a radial variable ``y``.  For a physical
    soliton of mass ``m`` the density scales according to

    ``ρ(r) = α(m)^2 ψ(√α(m) r)^2``,

    where ``α(m) = (m / mass0)^2`` and ``mass0`` is the normalisation
    stored in the soliton file.  Outside the support of ``ψ`` the
    density is zero.  This scaling law follows directly from the
    Gross–Pitaevskii equations in rescaled units and is implemented
    verbatim from the reference Mathematica code.

    Parameters
    ----------
    m : float
        Physical soliton mass.  Must be positive.

    sol : dict
        Dictionary returned by :func:`load_soliton_file`.  Must contain
        keys ``'ymax'``, ``'psi'`` and ``'alpha_func'``.

    Returns
    -------
    callable
        Function ``ρ(r)`` mapping a scalar or array of radii to the
        corresponding density.  Negative radii are clipped to zero.

    Raises
    ------
    ValueError
        If ``m`` is non‑positive or if the soliton dictionary lacks
        required keys.
    """
    if m <= 0:
        raise ValueError("Soliton mass must be positive.")
    if not all(k in sol for k in ("ymax", "psi", "alpha_func")):
        raise KeyError("The sol argument is missing required keys.")
    alpha = sol["alpha_func"](m)
    sqrt_alpha = np.sqrt(alpha)
    ymax = float(sol["ymax"])
    psi = sol["psi"]  # interpolation function ψ(y)
    def rho_of_r(r: np.ndarray | float) -> np.ndarray:
        # Vectorise input
        rr = np.asarray(r, dtype=np.float64)
        # Clip negative radii
        if np.any(rr < 0):
            warnings.warn(
                "soliton_density_function received negative r; clipping to zero.",
                RuntimeWarning,
            )
        rr_pos = np.maximum(rr, 0.0)
        # Compute scaled coordinate x = √α r
        x = sqrt_alpha * rr_pos
        # Evaluate ψ(x) where x ≤ ymax; else zero
        psi_vals = psi(x)
        # Mask values outside [0,ymax] (interp returns 0 for NaN)
        mask = (x <= ymax)
        rho_vals = np.zeros_like(rr_pos)
        rho_vals[mask] = (alpha ** 2) * (psi_vals[mask] ** 2)
        # Points with x > ymax remain zero
        return rho_vals
    return rho_of_r


def compute_potential(
    r_grid: np.ndarray,
    rho_func: Callable[[np.ndarray], np.ndarray],
    *,
    zero_at_rmax: bool = False,
) -> np.ndarray:
    """Compute the Newtonian potential Φ(r) from a spherically symmetric density.

    The radial Poisson equation in spherical symmetry reads

        u''(r) = 4π r ρ(r),    u(0) = 0,    u(r_max) = -M,

    where ``M = ∫_0^{r_max} 4π r^2 ρ(r) dr`` is the total mass enclosed
    within the computational domain.  The Newtonian potential is then
    given by ``Φ(r) = u(r) / r``.  By default, the outer boundary condition
    follows the reference implementation ``u(r_max) = -M``, which yields
    ``Φ(r_max) = -M/r_max`` and matches the exterior point‑mass solution of
    mass ``M``.

    Optional argument
    ----------
    zero_at_rmax : bool, optional
    When ``True``, enforce the boundary condition ``u(r_max) = 0`` instead
    of the default. This is equivalent to adding a constant ``M/r_max`` to
    the potential ``Φ``: the force field and the eigenfunctions are
    unchanged, and all eigenvalues shift upward by the same constant
    ``M/r_max``. Default ``False`` (keep ``u(r_max) = -M``).

    Parameters
    ----------
    r_grid : np.ndarray
        One‑dimensional radial grid on which to compute ``Φ``.  Must be
        strictly increasing with at least two points.  The largest
        element ``r_grid[-1]`` defines ``r_max``.

    rho_func : callable
        Function mapping radii to densities.  Should broadcast over
        vector inputs.

    Returns
    -------
    np.ndarray
        Array of potential values ``Φ(r)`` of the same length as
        ``r_grid``.

    Raises
    ------
    ValueError
        If the grid is not increasing or contains too few points.
    """
    r = np.asarray(r_grid, dtype=np.float64)
    n = len(r)
    if n < 2:
        raise ValueError("compute_potential requires at least two grid points.")
    if not np.all(np.diff(r) > 0):
        raise ValueError("r_grid must be strictly increasing.")
    # Compute the mass enclosed up to r_max
    # Compute the density on the grid.  This call should broadcast
    # naturally over arrays.
    rho_vals = rho_func(r)
    # Approximate the total mass enclosed within r_max.  The integral
    # ∫ 0^r_max 4π r^2 ρ(r) dr is computed via the trapezoidal rule.
    integrand_mass = 4.0 * np.pi * r ** 2 * rho_vals
    mass = np.trapz(integrand_mass, r)
    # The differential equation u''(r) = 4π r ρ(r) has general solution
    # u(r) = ∫_0^r (r - s) 4π s ρ(s) ds + C r.  Here we compute the
    # particular integral using an O(n) algorithm.  Let s_j = 4π r_j ρ_j
    # and dr the uniform spacing.  Then
    #   u0[k] = r_k * Σ_{j≤k} s_j dr - Σ_{j≤k} s_j r_j dr
    # This is equivalent to the double integral and is far more
    # efficient than nested loops or cumulative_trapezoid calls.
    # Compute source term s_j
    s_vals = 4.0 * np.pi * r * rho_vals
    # Uniform spacing h
    h = r[1] - r[0]
    # Cumulative sums of s_j and s_j*r_j multiplied by dr
    cum_s = np.cumsum(s_vals) * h
    cum_sr = np.cumsum(s_vals * r) * h
    # Particular solution u0
    u0 = r * cum_s - cum_sr
    # Add homogeneous solution C r.  Boundary condition at r_max depends on zero_at_rmax.
    r_max = r[-1]
    if r_max <= 0:
        raise ValueError("The radial grid must extend to positive r_max.")
    # Choose C to satisfy the desired outer boundary condition:
    # - Default (zero_at_rmax=False):   u(r_max) = -M  => C = (-M - u0(r_max)) / r_max
    # - Alternative (zero_at_rmax=True): u(r_max) = 0   => C = (0 - u0(r_max)) / r_max
    if zero_at_rmax:
        C = (-u0[-1]) / r_max
    else:
        C = (-mass - u0[-1]) / r_max
    u = u0 + C * r
    # Compute Φ(r) = u(r) / r with a proper r→0 limit.  If the grid
    # includes r=0, use the one‑sided derivative across the first
    # interval:
    #   Φ(0) = lim_{r→0} u(r)/r = u'(0) ≈ (u[1] - u[0]) / (r[1] - r[0]).
    # For r>0, use the usual ratio.
    phi = np.empty_like(r)
    phi[1:] = u[1:] / r[1:]
    phi[0] = (u[1] - u[0]) / (r[1] - r[0]) if r[0] == 0.0 else (u[0] / r[0])
    return phi


def compute_potential_from_grid(
    r_grid: np.ndarray,
    potential_path: str,
    *,
    com_path: Optional[str] = None,
    dx: Optional[float] = None,
    assume_centered: bool = False,
    method: str = "spherical_average",
    direction: Optional[Any] = None,
    strict: bool = True,
    subtract_mean: bool = False,
) -> np.ndarray:
    """Reconstruct a radial gravitational potential profile Φ(r) from a 3‑D grid file.

    This function is an *alternative* to :func:`compute_potential` that produces a
    potential array ``phi`` sampled on the supplied radial grid ``r_grid`` but
    does **not** solve the Poisson equation on the fly.  Instead it loads a
    precomputed three‑dimensional potential (e.g. a simulation output file
    ``G3D_#000.npy``), optionally recentres the field using a centre‑of‑mass
    (COM) position file (``UCM_#000.npy``), performs a spherical average, and
    interpolates the resulting radial profile onto ``r_grid``.  The returned
    array is therefore *format compatible* with the output of
    :func:`compute_potential` and can be used as a drop‑in replacement in the
    eigenvalue workflow (e.g. ``phi = compute_potential_from_grid(r_grid, ...)``).

    Parameters
    ----------
    r_grid : np.ndarray
        One‑dimensional strictly increasing radial grid (must exclude ``r=0``)
        at which to evaluate the radial potential.  Typically the same grid used
        later for the finite‑difference eigenvalue solve.

    potential_path : str
        Path to a NumPy ``.npy`` file containing a cubic 3‑D potential array
        ``phi_grid`` of shape ``(N,N,N)`` in *code units*.  The grid is assumed
        to be sampled on cell centres spanning ``[-L/2, +L/2]`` in each
        coordinate with uniform spacing ``dx``.  If ``dx`` is not supplied it is
        inferred as ``dx ≈ 2*max(r_grid)/N`` (i.e. the radial grid's outer edge
        is assumed to lie inside the simulation cube of half‑length ``L/2``).

    com_path : str, optional
        Path to a ``UCM_#NNN.npy`` file with the COM coordinates ``(cx, cy, cz)``
        in the same length units as the potential grid.  When provided (and
        ``assume_centered`` is ``False``) the 3‑D potential is *recentred* by an
        integer voxel roll so that the COM coincides with the origin prior to
        spherical averaging.  Missing or unreadable COM files fall back to a
        zero shift with a warning.

    dx : float, optional
        Grid spacing of the 3‑D potential.  If ``None`` it is inferred as
        described above.  Providing ``dx`` avoids the small inference error that
        can arise when the simulation cube is slightly larger than ``2*r_max``.

    assume_centered : bool, optional
        If ``True`` the potential is assumed already COM‑centred and no rolling
        shift is applied even if ``com_path`` is given.  Defaults to ``False``.

    method : {'spherical_average', 'line_cut'}
        Reduction strategy.  ``'spherical_average'`` computes a shell average
        and interpolates to ``r_grid`` (legacy behaviour).  ``'line_cut'``
        samples the 3‑D field along a straight line through the origin with
        trilinear interpolation and returns the values at positions
        ``x = r * \hat{d}``, where ``\hat{d}`` is a user‑provided unit
        direction vector (see ``direction``).  The spherical‑average behaviour
        is unchanged.

    direction : {'x','y','z'} or sequence of 3 floats, optional
        Line‑cut direction when ``method='line_cut'``.  Accepts the strings
        ``'x'``, ``'y'``, ``'z'`` (and their negated forms ``'-x'`` etc.) or a
        three‑component vector.  The vector is normalised internally; a zero
        vector raises ``ValueError``.  Ignored for other methods.

    strict : bool, optional
        If ``True`` (default) raise a ``ValueError`` when ``r_grid`` extends
        beyond the reliable spherical average radius (i.e. the largest shell
        fully contained inside the cube).  If ``False`` the outer values are
        clamped to the last valid shell average.

    subtract_mean : bool, optional
        If ``True`` subtract the mean of the 3‑D potential *after* COM recentering
        *before* spherical averaging.  Some simulation pipelines define the
        potential only up to an additive constant (e.g. k=0 FFT mode removed);
        removing the mean can simplify comparisons.  By default no constant is
        removed so that the raw profile is preserved.

    Returns
    -------
    np.ndarray
        Array ``phi`` of shape ``(len(r_grid),)`` with the interpolated radial
        potential.  dtype is ``float64``.  The mapping of entries matches that
        of :func:`compute_potential` and therefore downstream routines expecting
        a potential sampled on ``r_grid`` can consume it transparently.

    Notes
    -----
    1. The interpolation uses 1‑D linear interpolation (``np.interp``) onto the
       target grid; higher‑order schemes are unnecessary given the typical grid
       resolution but can be added later if needed.
    2. Because the COM shift is performed by integer voxel rolling, any sub‑cell
       fractional COM offset is intentionally *ignored* (consistent with the
       reference notebook's diagnostic cell).  This keeps the operation
       inexpensive and avoids introducing an additional interpolation layer.
    3. The outermost reliable radius is taken to be ``r_max_shell = L/2 - 0.5*dx``
       (the centre of the outermost voxel shell that still fits entirely within
       the cube).  Requesting radii beyond this threshold either raises or is
       clamped depending on ``strict``.
    4. No deprojection or Poisson solve is performed; the returned data merely
       reflects the supplied snapshot.
    """
    r = np.asarray(r_grid, dtype=np.float64)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("r_grid must be a one‑dimensional array of length >= 2.")
    if not np.all(np.diff(r) > 0):
        raise ValueError("r_grid must be strictly increasing and exclude r=0.")

    # Load 3‑D potential grid
    try:
        phi_grid = np.load(potential_path).astype(np.float64)
    except Exception as e:  # pragma: no cover - IO errors are runtime dependent
        raise IOError(f"Failed to load potential grid from '{potential_path}': {e}")
    if phi_grid.ndim != 3 or phi_grid.shape[0] != phi_grid.shape[1] or phi_grid.shape[1] != phi_grid.shape[2]:
        raise ValueError("Potential grid must be a cubic 3‑D array (N,N,N).")
    N = phi_grid.shape[0]

    # Infer / validate grid spacing.
    if dx is None:
        # Assume r_max is close to half the cube length => L ≈ 2 * r[-1]
        # Guard against zero division.
        r_max_target = float(r[-1])
        if r_max_target <= 0:
            raise ValueError("r_grid must extend to positive radii.")
        dx = (2.0 * r_max_target) / N
    dx = float(dx)
    if dx <= 0:
        raise ValueError("dx must be positive.")
    L = dx * N

    # Optional COM recentering (integer‑voxel roll).
    if not assume_centered and com_path:
        try:
            com = np.load(com_path)
            if com.size < 3:
                warnings.warn(
                    f"COM file '{com_path}' does not contain three coordinates; skipping shift.",
                    RuntimeWarning,
                )
            else:
                cx, cy, cz = map(float, com[:3])
                shift_i = int(np.rint(cx / dx))
                shift_j = int(np.rint(cy / dx))
                shift_k = int(np.rint(cz / dx))
                # Roll negative shifts so that COM moves to origin.
                phi_grid = np.roll(np.roll(np.roll(phi_grid, -shift_i, axis=0), -shift_j, axis=1), -shift_k, axis=2)
        except FileNotFoundError:
            warnings.warn(
                f"COM path '{com_path}' not found; proceeding without recentring.",
                RuntimeWarning,
            )
        except Exception as e:  # pragma: no cover - defensive
            warnings.warn(
                f"Failed to apply COM shift from '{com_path}': {e}; proceeding without shift.",
                RuntimeWarning,
            )

    # Optionally subtract the mean (constant offset ambiguity in FFT Poisson solves).
    if subtract_mean:
        phi_grid = phi_grid - float(np.mean(phi_grid))

    # Build coordinate array of cell centres: (i+0.5)*dx - L/2
    ax = (np.arange(N, dtype=np.float64) + 0.5) * dx - 0.5 * L
    # Squared coordinates (broadcast friendly)
    X2 = ax[:, None, None] ** 2
    Y2 = ax[None, :, None] ** 2
    Z2 = ax[None, None, :] ** 2
    R = np.sqrt(X2 + Y2 + Z2)

    # Helper for trilinear interpolation at a single point (x,y,z)
    def _interp_trilinear_at(x: float, y: float, z: float) -> float:
        ax0 = (x + 0.5 * L) / dx - 0.5
        ay0 = (y + 0.5 * L) / dx - 0.5
        az0 = (z + 0.5 * L) / dx - 0.5
        if not (0.0 <= ax0 <= (N - 1) and 0.0 <= ay0 <= (N - 1) and 0.0 <= az0 <= (N - 1)):
            return float("nan")
        i0, j0, k0 = int(np.floor(ax0)), int(np.floor(ay0)), int(np.floor(az0))
        i1, j1, k1 = min(i0 + 1, N - 1), min(j0 + 1, N - 1), min(k0 + 1, N - 1)
        dxu, dyu, dzu = ax0 - i0, ay0 - j0, az0 - k0
        c000 = phi_grid[i0, j0, k0]; c100 = phi_grid[i1, j0, k0]
        c010 = phi_grid[i0, j1, k0]; c110 = phi_grid[i1, j1, k0]
        c001 = phi_grid[i0, j0, k1]; c101 = phi_grid[i1, j0, k1]
        c011 = phi_grid[i0, j1, k1]; c111 = phi_grid[i1, j1, k1]
        c00 = c000 * (1 - dxu) + c100 * dxu
        c01 = c001 * (1 - dxu) + c101 * dxu
        c10 = c010 * (1 - dxu) + c110 * dxu
        c11 = c011 * (1 - dxu) + c111 * dxu
        c0 = c00 * (1 - dyu) + c10 * dyu
        c1 = c01 * (1 - dyu) + c11 * dyu
        return float(c0 * (1 - dzu) + c1 * dzu)

    if method == "line_cut":
        # Parse direction -> unit vector
        if direction is None:
            # Default to +x to mirror modal profile computation x=r, y=z=0
            dvec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif isinstance(direction, str):
            s = direction.strip().lower()
            sign = -1.0 if s.startswith("-") else 1.0
            base = s.lstrip("-")
            if base == "x":
                dvec = np.array([sign, 0.0, 0.0], dtype=np.float64)
            elif base == "y":
                dvec = np.array([0.0, sign, 0.0], dtype=np.float64)
            elif base == "z":
                dvec = np.array([0.0, 0.0, sign], dtype=np.float64)
            else:
                raise ValueError("direction string must be one of {'x','y','z','-x','-y','-z'}")
        else:
            arr = np.asarray(direction, dtype=np.float64).reshape(-1)
            if arr.size != 3:
                raise ValueError("direction must be a 3‑component vector.")
            dvec = arr
        norm = float(np.linalg.norm(dvec))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("direction must be non‑zero and finite.")
        d_hat = dvec / norm
        # Maximum usable radius before leaving cube (inclusive of outer cell centres)
        amax = 0.5 * L - 0.5 * dx  # = ax[-1]
        comp = np.abs(d_hat)
        # For zero components, ignore the constraint by using +inf
        with np.errstate(divide="ignore", invalid="ignore"):
            r_cap_per_axis = np.where(comp > 0, amax / comp, np.inf)
        r_cap = float(np.min(r_cap_per_axis))
        if float(r[-1]) > r_cap + 1e-12:
            msg = (
                f"r_grid extends to {r[-1]:.6g} which exceeds line‑cut limit {r_cap:.6g} for direction {d_hat}"
            )
            if strict:
                raise ValueError(msg)
            warnings.warn(msg + "; clamping outer values.", RuntimeWarning)
        # Evaluate along the line; clamp beyond r_cap to last valid value if strict=False
        phi_line = np.empty_like(r, dtype=np.float64)
        last_val: Optional[float] = None
        for i, rr in enumerate(r):
            if rr <= r_cap + 1e-15:
                x, y, z = (d_hat * rr).tolist()
                val = _interp_trilinear_at(x, y, z)
                if not np.isfinite(val):
                    # Shouldn't happen inside r_cap, but guard anyway
                    if strict:
                        raise RuntimeError("Interpolation failed within valid domain.")
                    val = last_val if last_val is not None else float("nan")
                phi_line[i] = val
                last_val = val
            else:
                # Out of bounds: clamp if allowed
                if strict:
                    raise ValueError("Encountered out‑of‑bounds r despite earlier check.")
                phi_line[i] = last_val if last_val is not None else float("nan")
        return phi_line

    if method != "spherical_average":
        raise NotImplementedError(f"method='{method}' is not implemented.")

    # Radial binning: dr = dx, only shells fully inside cube are considered.
    r_max_shell = 0.5 * L - 0.5 * dx  # centre of outermost intact shell
    dr = dx
    bins = np.arange(0.0, r_max_shell + dr, dr)
    flat_r = R.ravel()
    flat_phi = phi_grid.ravel()
    counts, edges = np.histogram(flat_r, bins=bins)
    sums, _ = np.histogram(flat_r, bins=bins, weights=flat_phi)
    with np.errstate(invalid="ignore", divide="ignore"):
        shell_avg = sums / np.maximum(1, counts)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    valid = counts > 0
    r_mid_v = r_mid[valid]
    shell_v = shell_avg[valid]
    if r_mid_v.size < 2:
        raise RuntimeError("Insufficient radial shells to construct a profile.")

    # Ensure target grid lies within reliable domain.
    if float(r[-1]) > float(r_mid_v[-1]) + 1e-12:
        msg = (
            f"r_grid extends to {r[-1]:.6g} which exceeds available spherical average radius "
            f"{r_mid_v[-1]:.6g}."
        )
        if strict:
            raise ValueError(msg + " Set strict=False to clamp to outer value.")
        warnings.warn(msg + " Clamping outer values.", RuntimeWarning)

    # Interpolate (linear) onto r_grid.  For radii beyond last shell (strict=False) clamp.
    phi_rad = np.interp(r, r_mid_v, shell_v, left=shell_v[0], right=shell_v[-1]).astype(np.float64)
    return phi_rad


def make_d2_matrix(r_grid: np.ndarray) -> _sparse.csr_matrix:
    """Construct the second derivative matrix with Dirichlet boundaries.

    The finite‑difference stencil used here mirrors the reference
    implementation in ``ULDMEig.m``.  On a uniform grid with spacing
    ``h`` the central difference approximation to the second derivative
    reads

      u''(r_i) ≈ (u_{i-1} - 2 u_i + u_{i+1}) / h^2.

    At the boundaries we assume that the wavefunction vanishes outside
    the domain (Dirichlet conditions).  This is implemented by simply
    omitting couplings beyond the first and last interior points.  No
    special one‑sided stencil is used; the same coefficients apply for
    all rows, and contributions from ghost nodes are implicitly zero.

    Parameters
    ----------
    r_grid : np.ndarray
        One‑dimensional radial grid.  Must be uniformly spaced with at
        least two points.  The values themselves are not used other
        than to infer the grid spacing ``h = r_grid[1] - r_grid[0]``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse tridiagonal matrix of shape ``(n,n)`` representing
        ``u''``.  Multiplying this matrix by a vector approximates
        ``d²u/dr²`` subject to Dirichlet boundary conditions at both
        ends.

    Raises
    ------
    ValueError
        If the grid is not uniform or contains fewer than two points.
    """
    r = np.asarray(r_grid, dtype=np.float64)
    n = len(r)
    if n < 2:
        raise ValueError("make_d2_matrix requires at least two grid points.")
    # Infer grid spacing and check uniformity
    h = r[1] - r[0]
    if not np.allclose(np.diff(r), h, rtol=1e-12, atol=1e-12):
        raise ValueError("The radial grid must have constant spacing.")
    # Main and off diagonals
    main_diag = np.full(n, -2.0 / h**2, dtype=np.float64)
    off_diag = np.full(n - 1, 1.0 / h**2, dtype=np.float64)
    # Construct tridiagonal matrix using scipy.sparse.diags
    D2 = _sparse.diags([
        off_diag, main_diag, off_diag
    ], offsets=[-1, 0, 1], shape=(n, n), format="csr")
    return D2


def make_h_matrix(
    r_grid: np.ndarray, phi: np.ndarray, ell: int
) -> Tuple[_sparse.csr_matrix, Dict[str, Any]]:
    """Assemble the radial Hamiltonian for a given angular momentum.

    Parameters
    ----------
    r_grid : np.ndarray
        Uniform radial grid (shape ``(n,)``).

    phi : np.ndarray
        Gravitational potential evaluated on ``r_grid``.  Shape and dtype
        must match ``r_grid``.

    ell : int
        Orbital angular momentum number.  Must be non‑negative.  The
        centrifugal term added to the potential is ``ell*(ell+1)/(2
        r_grid^2)``.

    Returns
    -------
    H : scipy.sparse.csr_matrix
        Sparse Hamiltonian of shape ``(n,n)`` acting on the reduced
        radial wavefunction ``u``.

    meta : dict
        Metadata containing the grid spacing ``h`` and any flags used
        during construction.  The dictionary keys are purely
        informational.

    Raises
    ------
    ValueError
        If ``ell`` is negative or the grid/potential lengths do not
        match.
    """
    if ell < 0 or int(ell) != ell:
        raise ValueError("ell must be a non‑negative integer in make_h_matrix.")
    r = np.asarray(r_grid, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if r.shape != phi.shape:
        raise ValueError("r_grid and phi must have identical shapes.")
    # Construct kinetic term
    D2 = make_d2_matrix(r)
    # Uniform spacing
    h = r[1] - r[0]
    # Centrifugal term ℓ(ℓ+1)/(2 r^2).  Use a small cutoff to avoid
    # division by zero when r is extremely close to zero.  Because the
    # reference grid excludes r=0 exactly, this is merely a safety net.
    tiny = 1e-20
    r_safe = np.where(np.abs(r) < tiny, tiny, r)
    cent = ell * (ell + 1) / (2.0 * r_safe**2)
    pot = phi + cent
    V = _sparse.diags(pot, offsets=0, format="csr")
    H = (-0.5) * D2 + V
    meta: Dict[str, Any] = {
        "h": h,
        "ell": ell,
        "n_points": len(r),
    }
    return H, meta


def solve_h(
    H: _sparse.csr_matrix,
    r_grid: np.ndarray,
    n_eig: int,
    shift: Optional[float] = None,
    maxiter: Optional[int] = None,
    tol: float = 1e-16,
    ell: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for the lowest ``n_eig`` eigenpairs of the radial Hamiltonian.

    This routine diagonalises a symmetric sparse matrix ``H`` and
    returns its lowest eigenvalues and corresponding eigenvectors.
    Because the radial Hamiltonian is Hermitian and typically
    indefinite (i.e., eigenvalues can be positive or negative), we use
    ``scipy.sparse.linalg.eigsh`` with the ``which='SA'`` option to
    select the ``n_eig`` smallest algebraic eigenvalues.  Optionally a
    spectral shift ``sigma`` can be supplied to perform shift‑invert
    mode, which often accelerates convergence for excited states.

    After solving the eigenproblem in the reduced basis ``u(r)``, we
    normalise each eigenvector with respect to the discrete L2 norm and
    convert it to the physical radial wavefunction ``f(r) = u(r) / r``.
    Near the inner boundary we enforce the regular series behaviour
    ``f(r) ∝ r^ℓ`` so that ``f`` remains finite and smooth even when the
    grid starts extremely close to ``r = 0``.  The eigenvalues and
    eigenvectors are returned sorted in ascending order.

    Parameters
    ----------
    H : scipy.sparse.csr_matrix
        The Hamiltonian matrix built by :func:`make_h_matrix`.

    r_grid : np.ndarray
        Radial grid used to convert ``u`` to ``f``.  Shape must be
        compatible with ``H.shape[0]``.

    n_eig : int
        Number of eigenpairs to compute.  Must be less than ``H.shape[0]``.

    shift : float, optional
        Spectral shift for the shift‑invert solver.  If ``None`` the
        standard solver is used with ``which='SA'``.  Use this when
        computing highly excited states to guide the solver towards the
        desired part of the spectrum.

    maxiter : int, optional
        Maximum number of iterations for the eigensolver.  If ``None``
        SciPy chooses a default based on the matrix size.

    tol : float, optional
        Convergence tolerance for the eigensolver.  Defaults to
        ``1e-10``.

    ell : int, optional
        Orbital angular momentum of the channel associated with the
        Hamiltonian ``H``.  Used to regularise ``f(r)`` near the origin
        so that ``f(r)`` obeys the analytic behaviour ``f(r) ∝ r^ℓ``.

    Returns
    -------
    E : np.ndarray
        Array of eigenvalues of length ``n_eig``, sorted in ascending
        order.

    f : np.ndarray
        2D array of shape ``(n_points, n_eig)`` where each column is
        the normalised physical radial wavefunction ``f(r)``.  The
        discrete L2 norm of each column (i.e., sum of squared
        magnitudes times the grid spacing) equals one.

    Raises
    ------
    ValueError
        If ``n_eig`` is not positive or exceeds the matrix dimension.

    Notes
    -----
    Sign ambiguity: eigenvectors returned by the solver are defined up to
    an overall phase.  To match the reference Mathematica implementation
    the sign of each eigenfunction is fixed so that the component with
    the largest magnitude is positive.  Without this normalisation the
    sign of an eigenvector can flip unpredictably depending on solver
    internals (or on the number of eigenvalues requested).  This
    alignment makes the parity of the numerical eigenfunctions
    deterministic and consistent with ``ULDMEig.m``.
    """
    n = H.shape[0]
    if n_eig <= 0 or n_eig >= n:
        raise ValueError(
            f"n_eig must be positive and less than the matrix dimension {n}."
        )
    r = np.asarray(r_grid, dtype=np.float64)
    if r.shape[0] != n:
        raise ValueError("r_grid length must match the dimension of H.")
    if ell < 0 or int(ell) != ell:
        raise ValueError("ell must be a non-negative integer in solve_h.")
    ell_int = int(ell)

    # Choose eigensolver options
    which = "SA"  # smallest algebraic eigenvalues
    # Use shift‑invert if a shift is provided.  SciPy implements this
    # through the sigma parameter which instructs the solver to find
    # eigenvalues near sigma.  Shift‑invert tends to converge faster
    # when the desired eigenvalues are not the extremal ones.
    if shift is not None:
        sigma = shift
        which = "LM"  # after shift‑invert, largest magnitude of (A - sigma I)^{-1} corresponds to eigenvalues nearest sigma
    else:
        sigma = None
    # If the grid includes r=0, impose the proper homogeneous Dirichlet boundary
    # u(0)=0 by solving on the interior subspace (indices 1..n-1) and then
    # padding u0=0.  This preserves symmetry and removes spurious dependence on rmin.
    origin_included = (r[0] == 0.0)
    if origin_included:
        H_eff = H[1:, 1:]
        n_eff = H_eff.shape[0]
        if n_eig >= n_eff:
            raise ValueError(
                f"n_eig must be less than the interior dimension {n_eff} when r[0]=0."
            )
        E, u_vecs_eff = _sparse_linalg.eigsh(
            H_eff,
            k=n_eig,
            which=which,
            sigma=sigma,
            maxiter=maxiter,
            tol=tol,
        )
        # Pad a zero at the top to reconstruct the full vector u with u(0)=0
        u_vecs = np.zeros((n, n_eig), dtype=np.float64)
        u_vecs[1:, :] = u_vecs_eff
    else:
        # Compute eigenvalues and eigenvectors.  We request n_eig eigenpairs.
        E, u_vecs = _sparse_linalg.eigsh(
            H,
            k=n_eig,
            which=which,
            sigma=sigma,
            maxiter=maxiter,
            tol=tol,
        )
    # Sort eigenvalues and reorder eigenvectors
    idx = np.argsort(E)
    E_sorted = E[idx]
    U_sorted = u_vecs[:, idx]
    # The reference Mathematica implementation divides all eigenvectors by
    # sqrt(h) once and then divides by r to obtain f.  Because the
    # eigenvectors returned by eigsh are orthonormal with respect to
    # the Euclidean norm (∑|u|^2 = 1), dividing by sqrt(h) ensures
    # that the discrete mass integral ∑|ψ|^2 r^2 dr equals one for
    # the ground state and is consistent for excited states.  We
    # therefore apply a global scaling to all eigenvectors rather than
    # normalising each individually.  This preserves orthogonality and
    # matches the reference output exactly.
    h = r[1] - r[0]
    scale = 1.0 / np.sqrt(h)
    U_scaled = U_sorted * scale
    # Convert to f = u / r while regularising the inner grid so that the
    # analytic behaviour f ∝ r^ℓ is respected.  We first form the naive
    # ratio and then overwrite the innermost points with a Taylor fit in
    # u(r).

    def _regularise_small_r(
        ell_val: int, r_arr: np.ndarray, u_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stable small‑r reconstruction of f(r)=u/r that tolerates r[0]=0.

        Strategy:
        - Fit u(r) near the origin using a short Taylor series with a window
          controlled by the grid spacing h (not by r[0]).  Use dimensionless
          coordinates t=r/h to improve conditioning.
        - For ℓ=0, fit u ≈ a1 r + a3 r^3 and set f(r) ≈ a1 + a3 r^2 on the
          first few points, enforcing f(0)=a1 when r[0]=0.
        - For ℓ>0, fit u ≈ Σ c_k r^{ℓ+1+2k} (k=0..K−1) with K≤3 using scaled
          powers; then f=u/r ≈ Σ c_k r^{ℓ+2k}.  Enforce f(0)=0.
        """
        n_tot = r_arr.size
        if n_tot < 2:
            return np.arange(n_tot), np.zeros_like(u_arr)
        h_loc = r_arr[1] - r_arr[0]
        r_end = r_arr[-1]
        # Window independent of r[0] to avoid collapse when r[0]→0.
        r_cut = min(6.0 * abs(h_loc), 0.03 * abs(r_end))
        fit_idx = np.nonzero(r_arr <= r_cut)[0]
        if fit_idx.size < 4:
            fit_idx = np.arange(min(8, n_tot))
        # Drop r=0 from the fit to avoid a zero row in Vandermonde.
        fit_idx = fit_idx[r_arr[fit_idx] > 0.0]
        if fit_idx.size < 2:
            fit_idx = np.arange(1, min(8, n_tot))

        t = r_arr[fit_idx] / h_loc  # dimensionless radii
        u = u_arr[fit_idx]

        if ell_val == 0:
            # s-wave: PDE-consistent small-r reconstruction.
            # Use Schrödinger eqn -1/2 u'' + φ u = E u => u'' = 2(φ - E) u.
            # For u ≈ a1 r + a3 r^3, we have u'' ≈ 6 a3 r and thus a3/a1 ≈ (1/6) (u''/u).
            # Estimate alpha := a3/a1 from finite-difference u''/u over a short prefix
            # (excluding boundaries), then solve a linear LS for a1 in u ≈ a1 r (1 + alpha r^2).

            # Build a small index window near the inner boundary
            idx0 = np.nonzero(r_arr <= r_cut)[0]
            if idx0.size < 6:
                idx0 = np.arange(min(10, n_tot))
            # Indices where we can compute a central second derivative
            # and avoid r=0 for division by r in the model
            inner = idx0[(idx0 > 0) & (idx0 < n_tot - 1) & (r_arr[idx0] > 0.0)]
            if inner.size < 2:
                # Fallback: take the first few interior points
                inner = np.arange(1, min(6, n_tot - 1))
            # Finite-difference second derivative u'' ≈ (u[i-1] - 2u[i] + u[i+1]) / h^2
            h2 = h_loc * h_loc
            u_pp = (u_arr[inner - 1] - 2.0 * u_arr[inner] + u_arr[inner + 1]) / h2
            u_mid = u_arr[inner]
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha_i = 0.16666666666666666 * np.divide(u_pp, u_mid, out=np.zeros_like(u_mid), where=np.abs(u_mid) > 0)
            # Robust estimate of alpha: median over the window, clipped to reasonable range
            if alpha_i.size == 0:
                alpha = 0.0
            else:
                alpha = float(np.median(alpha_i))
                # Clip extreme values to avoid numerical blow-ups if u_mid is tiny
                alpha = float(np.clip(alpha, -1e6, 1e6))
            # Now solve for a1 in u ≈ a1 r (1 + alpha r^2) over the same inner window
            g = r_arr[inner] * (1.0 + alpha * (r_arr[inner] ** 2))
            # Least-squares a1 = argmin ||a1*g - u||
            denom = float(np.dot(g, g))
            if denom > 0:
                a1 = float(np.dot(g, u_arr[inner]) / denom)
            else:
                # Fallback to simple slope using the first two points
                if n_tot >= 2:
                    a1 = float(u_arr[1] / h_loc)
                else:
                    a1 = 0.0
            # Construct replacement over a short prefix
            eval_end = min(max(int(idx0[-1]) + 1, 6), n_tot)
            reg_idx = np.arange(eval_end)
            r_small = r_arr[reg_idx]
            f_small = a1 * (1.0 + alpha * (r_small ** 2))
            if r_arr[0] == 0.0 and reg_idx.size > 0:
                f_small[0] = a1
            return reg_idx, f_small

        # ℓ>0: fit u ≈ Σ c_k r^{ℓ+1+2k}, k=0..K−1 in scaled coordinates.
        n_terms = 3
        pow_u = (ell_val + 1) + 2 * np.arange(n_terms)
        # Solve for scaled coefficients: u ≈ Σ (c_k h^{p_k}) t^{p_k}
        A = (t[:, None]) ** pow_u[None, :]
        b = u
        try:
            coeffs_scaled, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            eval_end = min(max(int(fit_idx[-1]) + 1, 2 * n_terms), n_tot)
            reg_idx = np.arange(eval_end)
            return reg_idx, np.zeros(reg_idx.size, dtype=float)
        p = pow_u.astype(float)
        c = coeffs_scaled / (h_loc ** p)  # convert back to physical coefficients
        pow_f = (ell_val + 2 * np.arange(n_terms)).astype(float)
        eval_end = min(max(int(fit_idx[-1]) + 1, 2 * n_terms), n_tot)
        reg_idx = np.arange(eval_end)
        r_small = r_arr[reg_idx]
        f_small = (r_small[:, None] ** pow_f[None, :]) @ c
        if r_arr[0] == 0.0 and reg_idx.size > 0:
            f_small = f_small.copy()
            f_small[0] = 0.0  # regular solution has f(0)=0 for ℓ>0
        return reg_idx, f_small

    ratio = np.divide(
        U_scaled,
        r[:, None],
        out=np.zeros_like(U_scaled),
        where=r[:, None] != 0.0,
    )
    f_mat = ratio.copy()

    for j in range(U_scaled.shape[1]):
        u_col = U_scaled[:, j]
        try:
            reg_idx, f_small = _regularise_small_r(ell_int, r, u_col)
            if reg_idx.size:
                f_mat[reg_idx, j] = f_small
        except (np.linalg.LinAlgError, ValueError):
            f_mat[:, j] = ratio[:, j]
    # Fix the sign of each eigenfunction to make its largest-magnitude
    # component positive.  This replicates the signOf operation in
    # Mathematica: signOf[arr_] = Sign@First@TakeLargestBy[arr, Abs, 1].
    # Without this step the ordering or number of requested eigenvalues
    # can cause arbitrary sign flips.  Here we determine the index of
    # the maximum absolute value in each column and multiply by ±1
    # accordingly.  Note that this does not alter orthonormality.
    for j in range(f_mat.shape[1]):
        col = f_mat[:, j]
        # index of maximal absolute value
        idx_max = int(np.argmax(np.abs(col)))
        if col[idx_max] < 0:
            f_mat[:, j] = -col
    return E_sorted, f_mat


def make_all_eigs(
    r_grid: np.ndarray,
    phi: np.ndarray,
    lmax: int,
    n_eig: int,
    shift: Optional[float] = None,
    maxiter: Optional[int] = None,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """Compute eigenvalues and eigenfunctions for all ℓ up to a maximum.

    Parameters
    ----------
    r_grid : np.ndarray
        Radial grid.  Must be uniform and strictly increasing.

    phi : np.ndarray
        Gravitational potential at each grid point.

    lmax : int
        Maximum orbital angular momentum to compute.  All ℓ from 0 to
        ``lmax`` inclusive will be processed.

    n_eig : int
        Number of radial excitations per ℓ to compute.  Must satisfy
        ``n_eig < len(r_grid)``.

    shift, maxiter, tol : optional
        Passed through to :func:`solve_h`.

    Returns
    -------
    dict
        A dictionary mapping strings of the form ``'ell/<ℓ>'`` to
        another dictionary with keys ``'r'``, ``'E'`` and ``'f'``.  The
        ``'r'`` entry contains the radial grid (shared for all ℓ), the
        ``'E'`` entry contains the eigenvalues as a one‑dimensional
        array and the ``'f'`` entry contains a 2D array of shape
        ``(n_points, n_eig)`` with columns ``f_{nℓ}(r)``.  This data
        structure matches that of the reference HDF5 file
        ``eigs_ref.h5``.

    Raises
    ------
    ValueError
        If ``lmax`` or ``n_eig`` are invalid or if the Hamiltonian
        construction fails.
    """
    if lmax < 0 or int(lmax) != lmax:
        raise ValueError("lmax must be a non‑negative integer.")
    if n_eig <= 0 or n_eig >= len(r_grid):
        raise ValueError("n_eig must be positive and less than the grid length.")
    result: Dict[str, Any] = {}
    # Compute the minimum of the gravitational potential.  The reference
    # implementation uses this value as a spectral shift when solving
    # the eigenproblem to accelerate convergence and improve accuracy.
    phi_min = float(np.min(phi))
    for ell in range(lmax + 1):
        H, meta = make_h_matrix(r_grid, phi, ell)
        # Use the provided shift if not None; otherwise use phi_min as
        # the shift.  Passing a shift to solve_h triggers shift–invert
        # mode in the eigensolver.  This closely mirrors the FEAST
        # interval method used in the Mathematica code.
        eff_shift = shift if shift is not None else phi_min
        E, f = solve_h(
            H,
            r_grid,
            n_eig,
            shift=eff_shift,
            maxiter=maxiter,
            tol=tol,
            ell=ell,
        )
        # Store under hierarchical key
        key = f"ell/{ell}"
        result[key] = {
            "r": np.array(r_grid, dtype=np.float64),
            "E": np.array(E, dtype=np.float64),
            "f": np.array(f, dtype=np.float64),
        }
    return result


def save_eigs_to_h5(data: Dict[str, Any], h5_path: str) -> None:
    """Write eigenvalue data to an HDF5 file matching the reference schema.

    The output file will contain a group ``/ell`` with subgroups
    ``0``, ``1``, ... corresponding to each angular momentum channel.
    Each subgroup will have datasets ``r``, ``E`` and ``f`` storing the
    radial grid, eigenvalues and eigenfunctions, respectively.  If the
    file already exists it will be overwritten.

    Parameters
    ----------
    data : dict
        Data structure returned by :func:`make_all_eigs`.

    h5_path : str
        Path to the HDF5 file to create.
    """
    if h5py is None:
        # Emit a warning and skip writing the file.  Without h5py the
        # caller cannot persist HDF5 data.  The eigen data is still
        # available in memory.
        warnings.warn(
            "save_eigs_to_h5 skipped because h5py is not installed."
            f" Data not written to {h5_path}.",
            RuntimeWarning,
        )
        return
    with h5py.File(h5_path, "w") as f:
        ell_group = f.create_group("ell")
        for key, val in data.items():
            parts = key.split("/")
            if len(parts) != 2:
                continue
            ell_str = parts[1]
            grp = ell_group.create_group(ell_str)
            grp.create_dataset("r", data=val["r"], dtype="float64")
            grp.create_dataset("E", data=val["E"], dtype="float64")
            grp.create_dataset("f", data=val["f"], dtype="float64")

__all__ = [
    "load_soliton_file",
    "soliton_density_function",
    "compute_potential",
    "compute_potential_from_grid",
    "make_d2_matrix",
    "make_h_matrix",
    "solve_h",
    "make_all_eigs",
    "save_eigs_to_h5",
    "compare_with_reference",
]

def compare_with_reference(
    eigs_dict: Dict[str, Any],
    ref_path: str,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compare computed eigenpairs against a reference HDF5 file.

    This helper function reads a reference file (presumably ``eigs_ref.h5``)
    and computes diagnostic statistics comparing the supplied eigen data
    ``eigs_dict`` with the reference.  The data structure ``eigs_dict``
    must have the same layout as produced by :func:`make_all_eigs`: keys
    of the form ``'ell/<ℓ>'`` mapping to dictionaries with keys ``'r'``,
    ``'E'`` and ``'f'``.

    The comparison assumes a one‐to‐one correspondence between the
    indices of the Python eigenfunctions and those stored in the
    reference.  Both the Python solver and the reference file use
    zero‐based indexing for the radial quantum number ``n``: the
    ground state has ``n=0``, the first excited state has ``n=1``, and
    so forth.  Therefore eigenvalue/ eigenvector index ``n`` in
    Python corresponds directly to index ``n`` in the reference.  No
    additional offset is applied when aligning the datasets.

    For each angular momentum channel ℓ the following statistics are
    computed:

    - ``shape_match``: boolean indicating whether the radial grid and
      number of eigenpairs match between the datasets.
    - ``E_rel_err_max``: maximum relative error over the aligned
      eigenvalues.
    - ``f_L2_rel_err_max``: maximum relative L2 error between aligned
      eigenfunctions.  The L2 norm uses the discrete measure ``dr``
      inferred from the reference grid.
    - ``corr_min``: minimum absolute correlation between aligned
      eigenfunctions after fixing overall signs.  A correlation of
      ``1`` indicates perfect agreement up to sign.

    Parameters
    ----------
    eigs_dict : dict
        Data structure returned from :func:`make_all_eigs`.

    ref_path : str
        Path to the reference HDF5 file.  The file must have groups
        ``/ell/<ℓ>`` with datasets ``r``, ``E`` and ``f``.

    atol, rtol : float, optional
        Absolute and relative tolerances used when checking for equal
        radial grids.  Defaults to ``1e-12``.

    verbose : bool, optional
        If ``True`` print per-ℓ diagnostics during comparison.  The
        returned dictionary always contains the diagnostics regardless
        of this flag.

    Returns
    -------
    dict
        A nested dictionary keyed by ℓ (as a string) containing the
        statistics described above.  An additional top-level key
        ``'overall_pass'`` is included indicating whether all ℓ
        channels meet the acceptance thresholds described in the
        specification.

    Raises
    ------
    ImportError
        If ``h5py`` is not available.

    KeyError
        If the reference file does not contain the expected groups or
        datasets.
    """
    if h5py is None:
        raise ImportError(
            "h5py is required to compare against the reference HDF5 file."
        )
    results: Dict[str, Any] = {}
    overall_pass = True
    with h5py.File(ref_path, "r") as ref:
        if "ell" not in ref:
            raise KeyError("Reference file lacks group 'ell'.")
        ell_group = ref["ell"]
        for key, our_data in eigs_dict.items():
            # Extract ℓ from the key (expect 'ell/<ℓ>')
            parts = key.split("/")
            if len(parts) != 2:
                continue
            ell_str = parts[1]
            if ell_str not in ell_group:
                raise KeyError(f"Reference file missing data for ell={ell_str}.")
            grp = ell_group[ell_str]
            # Load reference datasets
            r_ref = np.asarray(grp["r"], dtype=np.float64)
            E_ref = np.asarray(grp["E"], dtype=np.float64)
            f_ref = np.asarray(grp["f"], dtype=np.float64)
            # Load our computed data
            r_py = np.asarray(our_data["r"], dtype=np.float64)
            E_py = np.asarray(our_data["E"], dtype=np.float64)
            f_py = np.asarray(our_data["f"], dtype=np.float64)
            # Check shapes
            shape_ok = (r_ref.shape == r_py.shape and f_ref.shape[0] == f_py.shape[0])
            # Ensure the grids match to within tolerance
            grids_match = shape_ok and np.allclose(r_ref, r_py, rtol=rtol, atol=atol)
            # Determine number of eigenfunctions to compare.  Python n range
            # 0..n_eig-1 corresponds directly to the reference.  We
            # require the reference to have at least n_eig states.
            n_eig = E_py.shape[0]
            if E_ref.shape[0] < n_eig:
                raise ValueError(
                    f"Reference has {E_ref.shape[0]} eigenvalues but at least {n_eig} are required."
                )
            # Energies: align indices without offset
            E_ref_aligned = E_ref[:n_eig]
            # Compute relative errors
            E_rel_err = np.abs(E_ref_aligned - E_py) / np.maximum(np.abs(E_ref_aligned), 1e-20)
            E_rel_err_max = float(np.max(E_rel_err))
            # Wavefunctions: align and compute correlation and L2 error
            f_ref_aligned = f_ref[:, :n_eig]
            f_py_aligned = f_py[:, :n_eig]
            # Determine grid spacing from reference
            dr = r_ref[1] - r_ref[0]
            corr_vals: List[float] = []
            rel_err_vals: List[float] = []
            for i in range(n_eig):
                f_ref_i = f_ref_aligned[:, i]
                f_py_i = f_py_aligned[:, i]
                # Compute correlation with L2 weight
                numerator = np.sum(f_ref_i * f_py_i) * dr
                norm_ref = np.sqrt(np.sum(f_ref_i ** 2) * dr)
                norm_py = np.sqrt(np.sum(f_py_i ** 2) * dr)
                # Avoid division by zero
                corr = 0.0
                if norm_ref > 0 and norm_py > 0:
                    corr = numerator / (norm_ref * norm_py)
                # Fix overall sign by making corr positive
                sign = 1.0
                if corr < 0:
                    sign = -1.0
                    corr = -corr
                    f_py_i = -f_py_i
                # Relative L2 error
                diff_norm = np.sqrt(np.sum((f_ref_i - f_py_i) ** 2) * dr)
                rel_err = diff_norm / (norm_ref + 1e-300)
                corr_vals.append(float(corr))
                rel_err_vals.append(float(rel_err))
            f_L2_rel_err_max = float(np.max(rel_err_vals)) if rel_err_vals else float("nan")
            corr_min = float(np.min(corr_vals)) if corr_vals else float("nan")
            # Acceptance thresholds
            pass_shape = grids_match and f_ref.shape[1] >= n_eig
            pass_E = E_rel_err_max <= 1e-5
            pass_f = f_L2_rel_err_max <= 1e-5 and corr_min >= 0.999
            channel_pass = pass_shape and pass_E and pass_f
            overall_pass = overall_pass and channel_pass
            results[ell_str] = {
                "shape_match": bool(pass_shape),
                "E_rel_err_max": E_rel_err_max,
                "f_L2_rel_err_max": f_L2_rel_err_max,
                "corr_min": corr_min,
                "pass": bool(channel_pass),
            }
            if verbose:
                print(
                    f"ℓ={ell_str}: shape_match={pass_shape}, "
                    f"E_rel_err_max={E_rel_err_max:.2e}, "
                    f"f_L2_rel_err_max={f_L2_rel_err_max:.2e}, "
                    f"corr_min={corr_min:.3f}, pass={channel_pass}"
                )
    results["overall_pass"] = bool(overall_pass)
    return results
