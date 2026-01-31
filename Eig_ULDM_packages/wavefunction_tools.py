"""Utilities for building and processing PyUL initial wavefunctions."""

from __future__ import annotations

from functools import partial
from typing import Dict, Iterable, List, Tuple

import os
import time
import numpy as np
import multiprocessing as mp

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

_BUILD_GLOBALS = {}
_MODE_MASS_GLOBALS = {}
_MODE_FRAC_GLOBALS = {}


def _sph_harm_ylm_pyshtools(l: int, m: int, theta, phi):
    import pyshtools as pysh

    return pysh.expand.spharm_lm(
        l, m, theta, phi, kind='complex', normalization='ortho', csphase=-1, degrees=False
    )


def _sph_harm_ylm_scipy(l: int, m: int, theta, phi):
    from scipy.special import sph_harm as _scipy_sph_harm

    return _scipy_sph_harm(m, l, phi, theta)


def _get_sph_harm_backend():
    """Return (sph_harm_ylm_func, backend_name)."""
    try:
        import pyshtools  # noqa: F401

        return _sph_harm_ylm_pyshtools, 'pyshtools'
    except Exception:
        return _sph_harm_ylm_scipy, 'scipy'


def _init_build_worker(
    gridvec: np.ndarray,
    com_code: np.ndarray,
    modes: list,
    epsilons: np.ndarray,
    radial_fn: dict,
    phase_const: float,
    vel_code: np.ndarray,
    vphase0: float,
    sph_harm_func,
):
    _BUILD_GLOBALS['gridvec'] = gridvec
    _BUILD_GLOBALS['com_code'] = com_code
    _BUILD_GLOBALS['modes'] = modes
    _BUILD_GLOBALS['epsilons'] = epsilons
    _BUILD_GLOBALS['radial_fn'] = radial_fn
    _BUILD_GLOBALS['phase_const'] = phase_const
    _BUILD_GLOBALS['vel_code'] = vel_code
    _BUILD_GLOBALS['vphase0'] = vphase0
    _BUILD_GLOBALS['sph_harm'] = sph_harm_func


def _compute_slice_worker(k: int):
    g = _BUILD_GLOBALS
    gridvec = g['gridvec']
    com_code = g['com_code']
    modes = g['modes']
    epsilons = g['epsilons']
    radial_fn = g['radial_fn']
    phase_const = g['phase_const']
    vel_code = g['vel_code']
    vphase0 = g['vphase0']
    sph_harm_ylm = g['sph_harm']

    z = gridvec[k]
    X, Y = np.meshgrid(gridvec, gridvec, indexing='ij')
    XR = X - com_code[0]
    YR = Y - com_code[1]
    ZR = z - com_code[2]

    r = np.sqrt(XR * XR + YR * YR + ZR * ZR)
    with np.errstate(invalid='ignore', divide='ignore'):
        theta = np.arccos(np.where(r > 0, ZR / r, 1.0))
    phi = np.mod(np.arctan2(YR, XR), 2 * np.pi)

    psi_slice = np.zeros((gridvec.size, gridvec.size), dtype=np.complex128)

    for idx, (n, l, m) in enumerate(modes):
        eps = epsilons[idx]
        if eps == 0:
            continue
        rg, fr = radial_fn[(l, n)]
        fr_vals = np.interp(r.ravel(), rg, fr, left=0.0, right=0.0).reshape(r.shape)
        Ylm = sph_harm_ylm(int(l), int(m), theta, phi)
        psi_slice += eps * fr_vals * Ylm

    if phase_const != 0.0:
        psi_slice *= np.exp(1j * phase_const)

    phase_plane = vel_code[0] * X + vel_code[1] * Y + vel_code[2] * z
    psi_slice *= np.exp(1j * (phase_plane + vphase0))

    return psi_slice


def _init_mode_mass_worker(
    gridvec: np.ndarray,
    com_code: np.ndarray,
    radial_fn: dict,
    l: int,
    n: int,
    m: int,
    eps: complex,
    sph_harm_func,
):
    _MODE_MASS_GLOBALS['gridvec'] = gridvec
    _MODE_MASS_GLOBALS['com_code'] = com_code
    _MODE_MASS_GLOBALS['radial_fn'] = radial_fn
    _MODE_MASS_GLOBALS['l'] = int(l)
    _MODE_MASS_GLOBALS['n'] = int(n)
    _MODE_MASS_GLOBALS['m'] = int(m)
    _MODE_MASS_GLOBALS['eps'] = eps
    _MODE_MASS_GLOBALS['sph_harm'] = sph_harm_func


def _mode_mass_slice_worker(k: int):
    g = _MODE_MASS_GLOBALS
    gridvec = g['gridvec']
    com_code = g['com_code']
    radial_fn = g['radial_fn']
    l = g['l']
    n = g['n']
    m = g['m']
    eps = g['eps']
    sph_harm_ylm = g['sph_harm']

    z = gridvec[k]
    X, Y = np.meshgrid(gridvec, gridvec, indexing='ij')
    XR = X - com_code[0]
    YR = Y - com_code[1]
    ZR = z - com_code[2]
    r = np.sqrt(XR * XR + YR * YR + ZR * ZR)
    with np.errstate(invalid='ignore', divide='ignore'):
        theta = np.arccos(np.where(r > 0, ZR / r, 1.0))
    phi = np.mod(np.arctan2(YR, XR), 2 * np.pi)
    rg, fr = radial_fn[(l, n)]
    fr_vals = np.interp(r.ravel(), rg, fr, left=0.0, right=0.0).reshape(r.shape)
    Ylm = sph_harm_ylm(int(l), int(m), theta, phi)
    psi_slice = eps * fr_vals * Ylm
    return float(np.sum(np.abs(psi_slice) ** 2))


def _init_mode_fraction_worker(
    path: str,
    rg: np.ndarray,
    fr: np.ndarray,
    l: int,
    m: int,
    gridvec: np.ndarray,
    com_code: np.ndarray,
    phi2d: np.ndarray,
    r_xy2: np.ndarray,
    sph_harm_func,
):
    _MODE_FRAC_GLOBALS['psi_mmap'] = np.load(path, mmap_mode='r')
    _MODE_FRAC_GLOBALS['rg'] = rg
    _MODE_FRAC_GLOBALS['fr'] = fr
    _MODE_FRAC_GLOBALS['l'] = int(l)
    _MODE_FRAC_GLOBALS['m'] = int(m)
    _MODE_FRAC_GLOBALS['gridvec'] = gridvec
    _MODE_FRAC_GLOBALS['com_code'] = com_code
    _MODE_FRAC_GLOBALS['phi2d'] = phi2d
    _MODE_FRAC_GLOBALS['r_xy2'] = r_xy2
    _MODE_FRAC_GLOBALS['sph_harm'] = sph_harm_func


def _mode_overlap_slice_worker(k: int):
    g = _MODE_FRAC_GLOBALS
    psi_mmap = g['psi_mmap']
    gridvec = g['gridvec']
    com_code = g['com_code']
    phi2d = g['phi2d']
    r_xy2 = g['r_xy2']
    rg = g['rg']
    fr = g['fr']
    l = g['l']
    m = g['m']
    sph_harm_ylm = g['sph_harm']

    z = gridvec[k]
    psi_slice = psi_mmap[:, :, k]
    ZR = z - com_code[2]
    r = np.sqrt(r_xy2 + ZR ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        theta = np.arccos(np.where(r > 0, ZR / r, 1.0))
    fr_vals = np.interp(r.ravel(), rg, fr, left=0.0, right=0.0).reshape(r.shape)
    Ylm = sph_harm_ylm(l, m, theta, phi2d)
    return np.sum(np.conjugate(fr_vals * Ylm) * psi_slice)


def build_initial_wavefunction(
    *,
    eigs_data: Dict,
    r_grid: np.ndarray,
    length: float,
    length_units: str,
    resol: int,
    CellCenteredGrid: bool,
    mS_code_unit: float,
    s_velocity_unit: str,
    nlm_list: Iterable[Tuple[int, int, int]],
    epsilon_nlm_list: Iterable[complex],
    center_of_mass: Iterable[float],
    COM_speed: Iterable[float],
    Phase: float,
    start_time: float,
    duration_units: str | None = None,
    init_basename: str = 'Init',
    nprocs: int = 8,
    run_root_dir: str | None = None,
):
    """Build a 3D initial wavefunction Psi and save it to disk.

    Returns a dictionary containing Psi, psi_path, COM_meas, M_tot, and mode_masses.
    """
    from .units import convert_between

    if run_root_dir is None:
        run_root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
    os.makedirs(run_root_dir, exist_ok=True)

    lengthC = convert_between(length, length_units, '', 'l')
    com_code = np.array(convert_between(np.array(center_of_mass), length_units, '', 'l'), dtype=float)
    vel_code = np.array(convert_between(np.array(COM_speed), s_velocity_unit, '', 'v'), dtype=float)
    if duration_units is not None:
        t0 = float(convert_between(start_time, duration_units, '', 't'))
    else:
        t0 = float(start_time)

    dx = float(lengthC) / float(resol)
    if CellCenteredGrid:
        gridvec = -lengthC / 2.0 + dx / 2.0 + np.arange(resol) * dx
    else:
        gridvec = np.linspace(-lengthC / 2.0, lengthC / 2.0, resol, endpoint=False)

    nlm_list = list(nlm_list)
    _unique_nl: List[Tuple[int, int]] = []
    for n, l, _ in nlm_list:
        if (l, n) not in _unique_nl:
            _unique_nl.append((l, n))

    radial_fn = {}
    for l, n in _unique_nl:
        key = f'ell/{int(l)}'
        if key not in eigs_data:
            raise KeyError(f"No eigen data for l={l}. Available: {[k for k in eigs_data if k.startswith('ell/')]}")
        f_mat = eigs_data[key]['f']
        if n < 0 or n >= f_mat.shape[1]:
            raise IndexError(f"For l={l}, requested n={n} is out of range (0..{f_mat.shape[1]-1})")
        radial_fn[(l, n)] = (r_grid.copy(), f_mat[:, n].copy())

    sph_harm_ylm, ylm_backend = _get_sph_harm_backend()

    modes = [(int(n), int(l), int(m)) for (n, l, m) in nlm_list]
    eps_arr = np.asarray(list(epsilon_nlm_list), dtype=np.complex128)
    if eps_arr.shape[0] != len(modes):
        raise ValueError("epsilon_nlm_list length must match nlm_list length")

    v2 = float(np.dot(vel_code, vel_code))
    vphase0 = -0.5 * v2 * t0

    Psi = np.zeros((resol, resol, resol), dtype=np.complex128)
    phase_const = float(Phase) * np.pi

    start_t = time.time()
    with mp.get_context('fork').Pool(
        processes=nprocs,
        initializer=_init_build_worker,
        initargs=(
            gridvec,
            com_code,
            modes,
            eps_arr,
            radial_fn,
            phase_const,
            vel_code,
            vphase0,
            sph_harm_ylm,
        ),
    ) as pool:
        iterator = pool.imap(_compute_slice_worker, range(resol), chunksize=1)
        if tqdm is not None:
            iterator = tqdm(
                enumerate(iterator),
                total=resol,
                desc=f"Building Psi [{ylm_backend}]",
                unit="slice",
            )
        else:
            iterator = enumerate(iterator)
        for k, sl in iterator:
            Psi[:, :, k] = sl

    Psi *= np.sqrt(float(mS_code_unit))

    rho = np.abs(Psi) ** 2
    Xv, Yv, Zv = np.meshgrid(gridvec, gridvec, gridvec, indexing='ij')
    vol = dx ** 3
    M_tot = float(np.sum(rho) * vol)
    if M_tot > 0:
        COM_meas = np.array(
            [
                float(np.sum(rho * Xv) * vol / M_tot),
                float(np.sum(rho * Yv) * vol / M_tot),
                float(np.sum(rho * Zv) * vol / M_tot),
            ]
        )
    else:
        COM_meas = np.array([np.nan, np.nan, np.nan])

    nz_modes = [(i, modes[i], eps_arr[i]) for i in range(len(modes)) if eps_arr[i] != 0]
    mode_masses = []
    if len(nz_modes) > 0:
        for i, (n_i, l_i, m_i), eps_i in nz_modes:
            with mp.get_context('fork').Pool(
                processes=nprocs,
                initializer=_init_mode_mass_worker,
                initargs=(
                    gridvec,
                    com_code,
                    radial_fn,
                    int(l_i),
                    int(n_i),
                    int(m_i),
                    eps_i,
                    sph_harm_ylm,
                ),
            ) as pool:
                ssum = 0.0
                iterator = pool.imap(_mode_mass_slice_worker, range(resol), chunksize=1)
                if tqdm is not None:
                    iterator = tqdm(
                        iterator,
                        total=resol,
                        desc=f"Mode mass (n={int(n_i)}, l={int(l_i)}, m={int(m_i)})",
                        unit="slice",
                        leave=False,
                    )
                for sval in iterator:
                    ssum += float(sval)
            mass_i = float(mS_code_unit) * ssum * vol
            mode_masses.append(((int(n_i), int(l_i), int(m_i)), mass_i))

        total_modes_mass = sum(m for (_, m) in mode_masses)
    else:
        total_modes_mass = 0.0

    psi_path = os.path.join(run_root_dir, f"{init_basename}_psi.npy")
    np.save(psi_path, Psi.astype(np.complex128, copy=False))

    print("Initial wavefunction generated.")
    print(f"  Backend for Y_lm: {ylm_backend}")
    print(f"  Grid: resol={resol}, CellCenteredGrid={CellCenteredGrid}, dx={dx:.6g} (code units)")
    print(f"  COM target (code units): {com_code}")
    print(f"  Measured COM (code units): {COM_meas}")
    print(f"  Total mass from |Psi|^2: {M_tot:.6e} (code units)")
    if len(nz_modes) > 0:
        print("  Per-mode mass contributions (code units) from non-zero epsilon modes:")
        for (n_i, l_i, m_i), mass_i in mode_masses:
            print(f"    (n={n_i}, l={l_i}, m={m_i}) -> {mass_i:.6e}")
        print(f"  Sum of per-mode masses (ignores interference): {total_modes_mass:.6e} (code units)")
    if len(nz_modes) > 1:
        interf_mass_diff = float(M_tot - total_modes_mass)
        print(f"  Interference mass (difference): {interf_mass_diff:.6e} (code units)")
    print(f"  Saved: {psi_path}")
    print(
        "Use this as InitPath in PyUL: \n  InitPath = '"
        + os.path.splitext(psi_path)[0]
        + "'  # without the _psi.npy suffix"
    )

    return {
        "Psi": Psi,
        "psi_path": psi_path,
        "COM_meas": COM_meas,
        "M_tot": M_tot,
        "mode_masses": mode_masses,
        "total_modes_mass": total_modes_mass,
        "ylm_backend": ylm_backend,
        "gridvec": gridvec,
        "dx": dx,
        "run_root_dir": run_root_dir,
        "elapsed_sec": time.time() - start_t,
    }


def deboost_wavefunction(
    *,
    init_basename: str,
    resol: int,
    length: float,
    length_units: str,
    CellCenteredGrid: bool,
    center_of_mass: Iterable[float] | None = None,
    COM_speed: Iterable[float] | None = None,
    s_velocity_unit: str | None = None,
    IsoP_like: bool = True,
    fd_order: int = 4,
    use_mask: bool = True,
    mask_mode: str = 'mass_fraction',
    mask_fraction: float = 1.0,
    density_frac: float = 1e-6,
    max_iters: int = 20,
    tol_kms: float = 1e-20,
    tol_pos_frac_dx: float = 0.05,
    psi_path: str | None = None,
):
    """Iteratively deboost Psi to match target COM and bulk velocity."""
    from .units import convert_between

    _out_lines: List[str] = []
    _add = lambda s: _out_lines.append(str(s))

    def _final_flush():
        if _out_lines:
            print("\n".join(_out_lines))

    if psi_path is None:
        base = str(init_basename)
        if base.endswith('.npy'):
            base = base[:-4]
        candidates = [base + '.npy', base + '_psi.npy']
        cwd = os.getcwd()
        paths_to_try = [p if os.path.isabs(p) else os.path.join(cwd, p) for p in candidates]
        psi_path = None
        for p in paths_to_try:
            if os.path.exists(p):
                psi_path = os.path.abspath(p)
                break
        if psi_path is None:
            _add("[Targeting] Could not locate the initial Psi file. Tried:")
            for p in paths_to_try:
                _add(f"  {p}")
            _final_flush()
            return {"psi_path": None, "deb_initpath": None, "deb_psipath": None}

    lengthC = float(convert_between(length, length_units, '', 'l'))
    N = int(resol)
    dx = lengthC / N
    if CellCenteredGrid:
        gridvec = -lengthC / 2 + dx / 2 + np.arange(N) * dx
    else:
        gridvec = np.linspace(-lengthC / 2, lengthC / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(gridvec, gridvec, gridvec, indexing='ij')
    dV = dx ** 3

    cm_target_len = list(center_of_mass) if center_of_mass is not None else [0.0, 0.0, 0.0]
    v_target_vel = list(COM_speed) if COM_speed is not None else [0.0, 0.0, 0.0]
    vel_units_name = str(s_velocity_unit) if s_velocity_unit is not None else 'km/s'

    com_tgt_code = np.array(convert_between(np.array(cm_target_len, dtype=float), length_units, '', 'l'), dtype=float)
    v_tgt_code = np.array(convert_between(np.array(v_target_vel, dtype=float), vel_units_name, '', 'v'), dtype=float)
    tol_pos_abs = float(tol_pos_frac_dx) * dx

    def _small_N_guard(n):
        return n < 5

    def grad_isolated_2(f, dx):
        df_dx = np.empty_like(f, dtype=f.dtype)
        df_dy = np.empty_like(f, dtype=f.dtype)
        df_dz = np.empty_like(f, dtype=f.dtype)
        df_dx[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * dx)
        df_dx[0, :, :] = (f[1, :, :] - f[0, :, :]) / dx
        df_dx[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dx
        df_dy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * dx)
        df_dy[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dx
        df_dy[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dx
        df_dz[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dx)
        df_dz[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dx
        df_dz[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dx
        return df_dx, df_dy, df_dz

    def grad_isolated_4(f, dx):
        if _small_N_guard(f.shape[0]) or _small_N_guard(f.shape[1]) or _small_N_guard(f.shape[2]):
            return grad_isolated_2(f, dx)
        df_dx = np.empty_like(f, dtype=f.dtype)
        df_dy = np.empty_like(f, dtype=f.dtype)
        df_dz = np.empty_like(f, dtype=f.dtype)
        df_dx[2:-2, :, :] = (-f[4:, :, :] + 8 * f[3:-1, :, :] - 8 * f[1:-3, :, :] + f[0:-4, :, :]) / (12 * dx)
        df_dx[0, :, :] = (-25 * f[0, :, :] + 48 * f[1, :, :] - 36 * f[2, :, :] + 16 * f[3, :, :] - 3 * f[4, :, :]) / (12 * dx)
        df_dx[1, :, :] = (f[2, :, :] - f[0, :, :]) / (2 * dx)
        df_dx[-2, :, :] = (f[-1, :, :] - f[-3, :, :]) / (2 * dx)
        df_dx[-1, :, :] = (25 * f[-1, :, :] - 48 * f[-2, :, :] + 36 * f[-3, :, :] - 16 * f[-4, :, :] + 3 * f[-5, :, :]) / (12 * dx)
        df_dy[:, 2:-2, :] = (-f[:, 4:, :] + 8 * f[:, 3:-1, :] - 8 * f[:, 1:-3, :] + f[:, 0:-4, :]) / (12 * dx)
        df_dy[:, 0, :] = (-25 * f[:, 0, :] + 48 * f[:, 1, :] - 36 * f[:, 2, :] + 16 * f[:, 3, :] - 3 * f[:, 4, :]) / (12 * dx)
        df_dy[:, 1, :] = (f[:, 2, :] - f[:, 0, :]) / (2 * dx)
        df_dy[:, -2, :] = (f[:, -1, :] - f[:, -3, :]) / (2 * dx)
        df_dy[:, -1, :] = (25 * f[:, -1, :] - 48 * f[:, -2, :] + 36 * f[:, -3, :] - 16 * f[:, -4, :] + 3 * f[:, -5, :]) / (12 * dx)
        df_dz[:, :, 2:-2] = (-f[:, :, 4:] + 8 * f[:, :, 3:-1] - 8 * f[:, :, 1:-3] + f[:, :, 0:-4]) / (12 * dx)
        df_dz[:, :, 0] = (-25 * f[:, :, 0] + 48 * f[:, :, 1] - 36 * f[:, :, 2] + 16 * f[:, :, 3] - 3 * f[:, :, 4]) / (12 * dx)
        df_dz[:, :, 1] = (f[:, :, 2] - f[:, :, 0]) / (2 * dx)
        df_dz[:, :, -2] = (f[:, :, -1] - f[:, :, -3]) / (2 * dx)
        df_dz[:, :, -1] = (25 * f[:, :, -1] - 48 * f[:, :, -2] + 36 * f[:, :, -3] - 16 * f[:, :, -4] + 3 * f[:, :, -5]) / (12 * dx)
        return df_dx, df_dy, df_dz

    def grad_periodic_2(f, dx):
        df_dx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)
        df_dy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)
        df_dz = (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2.0 * dx)
        return df_dx, df_dy, df_dz

    def grad_periodic_4(f, dx):
        df_dx = (-np.roll(f, -2, axis=0) + 8 * np.roll(f, -1, axis=0) - 8 * np.roll(f, 1, axis=0) + np.roll(f, 2, axis=0)) / (12.0 * dx)
        df_dy = (-np.roll(f, -2, axis=1) + 8 * np.roll(f, -1, axis=1) - 8 * np.roll(f, 1, axis=1) + np.roll(f, 2, axis=1)) / (12.0 * dx)
        df_dz = (-np.roll(f, -2, axis=2) + 8 * np.roll(f, -1, axis=2) - 8 * np.roll(f, 1, axis=2) + np.roll(f, 2, axis=2)) / (12.0 * dx)
        return df_dx, df_dy, df_dz

    grad_op = grad_isolated_4 if (IsoP_like and fd_order == 4) else None
    if grad_op is None:
        if IsoP_like:
            grad_op = grad_isolated_2
        else:
            grad_op = grad_periodic_4 if fd_order == 4 else grad_periodic_2

    def build_mask_around_COM(rho):
        M_full = float(np.sum(rho) * dV)
        if M_full <= 0:
            return None, (np.nan, np.nan, np.nan), 0.0
        COMx_full = float(np.sum(rho * X) * dV / M_full)
        COMy_full = float(np.sum(rho * Y) * dV / M_full)
        COMz_full = float(np.sum(rho * Z) * dV / M_full)
        if not use_mask:
            return None, (COMx_full, COMy_full, COMz_full), M_full
        XR = X - COMx_full
        YR = Y - COMy_full
        ZR = Z - COMz_full
        r = np.sqrt(XR * XR + YR * YR + ZR * ZR)
        if mask_mode == 'mass_fraction':
            flat_r = r.ravel()
            flat_m = (rho * dV).ravel()
            order = np.argsort(flat_r)
            cumM = np.cumsum(flat_m[order])
            target = mask_fraction * (cumM[-1] if cumM.size > 0 else 0.0)
            idx = np.searchsorted(cumM, target) if cumM.size > 0 else 0
            r_cut = float(flat_r[order[min(idx, len(order) - 1)]]) if len(order) > 0 else np.inf
            mask = (r <= r_cut)
            _add(f"[Targeting] Region mask: mass-fraction, fraction={mask_fraction:.6f}, r_cut={r_cut:.6e} (code length units).")
        else:
            rho_max = float(np.max(rho)) if rho.size > 0 else 0.0
            thr = rho_max * float(density_frac)
            mask = (rho >= thr)
            _add(f"[Targeting] Region mask: density-threshold, rho >= {density_frac:.3e} * rho_max (threshold={thr:.6e}).")
        return mask, (COMx_full, COMy_full, COMz_full), M_full

    def measure_MPv(Psi, mask):
        rho = np.abs(Psi) ** 2
        if mask is None:
            M = float(np.sum(rho) * dV)
            COMx = float(np.sum(rho * X) * dV / M) if M > 0 else np.nan
            COMy = float(np.sum(rho * Y) * dV / M) if M > 0 else np.nan
            COMz = float(np.sum(rho * Z) * dV / M) if M > 0 else np.nan
            dpsi_dx, dpsi_dy, dpsi_dz = grad_op(Psi, dx)
            jx = np.imag(np.conjugate(Psi) * dpsi_dx)
            jy = np.imag(np.conjugate(Psi) * dpsi_dy)
            jz = np.imag(np.conjugate(Psi) * dpsi_dz)
            Px = float(np.sum(jx) * dV)
            Py = float(np.sum(jy) * dV)
            Pz = float(np.sum(jz) * dV)
        else:
            M = float(np.sum((np.abs(Psi) ** 2)[mask]) * dV)
            COMx = float(np.sum((np.abs(Psi) ** 2 * X)[mask]) * dV / M) if M > 0 else np.nan
            COMy = float(np.sum((np.abs(Psi) ** 2 * Y)[mask]) * dV / M) if M > 0 else np.nan
            COMz = float(np.sum((np.abs(Psi) ** 2 * Z)[mask]) * dV / M) if M > 0 else np.nan
            dpsi_dx, dpsi_dy, dpsi_dz = grad_op(Psi, dx)
            jx = np.imag(np.conjugate(Psi) * dpsi_dx)
            jy = np.imag(np.conjugate(Psi) * dpsi_dy)
            jz = np.imag(np.conjugate(Psi) * dpsi_dz)
            Px = float(np.sum(jx[mask]) * dV)
            Py = float(np.sum(jy[mask]) * dV)
            Pz = float(np.sum(jz[mask]) * dV)
        if M <= 0:
            return 0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), (COMx, COMy, COMz)
        vx, vy, vz = Px / M, Py / M, Pz / M
        return M, Px, Py, Pz, (vx, vy, vz), (COMx, COMy, COMz)

    def translate_periodic(Psi, delta):
        kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        kz = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        phase = np.exp(-1j * (KX * delta[0] + KY * delta[1] + KZ * delta[2]))
        F = np.fft.fftn(Psi)
        return np.fft.ifftn(F * phase)

    def translate_isolated(Psi, delta):
        try:
            from scipy.ndimage import shift as ndi_shift
            shift_pix = (delta[0] / dx, delta[1] / dx, delta[2] / dx)
            return ndi_shift(Psi, shift=shift_pix, order=3, mode='constant', cval=0.0, prefilter=True)
        except Exception:
            _add("[Targeting] SciPy not available; skipping isolated translation (COM targeting may not converge).")
            return Psi

    translate_op = translate_isolated if IsoP_like else translate_periodic

    Psi0 = np.load(psi_path)
    if Psi0.shape != (N, N, N):
        _add(f"[Targeting] Shape mismatch: Psi.shape={Psi0.shape}, expected {(N, N, N)}")
        _final_flush()
        return {"psi_path": psi_path, "deb_initpath": None, "deb_psipath": None}

    Psi_curr = Psi0
    _add(f"[Targeting] Input file: {psi_path}")
    _add(f"  Grid: N={N}, dx={dx:.6e} (code length units)")
    _add(f"  Targets (code units): COM={com_tgt_code.tolist()}, v={v_tgt_code.tolist()}")
    _add(f"  Tolerances: |v - v_tgt| < {tol_kms:.3e} km/s, |COM - COM_tgt| < {tol_pos_abs:.3e} (code length units)")

    for it in range(1, int(max_iters) + 1):
        rho = np.abs(Psi_curr) ** 2
        mask, COM_full, M_full = build_mask_around_COM(rho)
        M_i, Px_i, Py_i, Pz_i, v_i_tuple, COM_i_tuple = measure_MPv(Psi_curr, mask)
        v_i_code = np.array([v_i_tuple[0], v_i_tuple[1], v_i_tuple[2]], dtype=float)
        v_i_kms = np.array(convert_between(v_i_code, '', 'km/s', 'v'), dtype=float)
        com_i_code = np.array([COM_i_tuple[0], COM_i_tuple[1], COM_i_tuple[2]], dtype=float)

        _add(f"  Iter {it}: v=[{v_i_kms[0]:.9e}, {v_i_kms[1]:.9e}, {v_i_kms[2]:.9e}] km/s; COM={com_i_code.tolist()} (code)")

        v_err_kms = float(np.linalg.norm(v_i_kms - np.array(convert_between(v_tgt_code, '', 'km/s', 'v'), dtype=float)))
        com_err = float(np.linalg.norm(com_i_code - com_tgt_code))
        if v_err_kms <= float(tol_kms) and com_err <= float(tol_pos_abs):
            _add(f"  Converged at iter {it}: |Δv|={v_err_kms:.6e} km/s, |ΔCOM|={com_err:.6e} (code)")
            break

        dv_code = v_tgt_code - v_i_code
        if np.linalg.norm(dv_code) > 0:
            phase_v = dv_code[0] * X + dv_code[1] * Y + dv_code[2] * Z
            Psi_curr = Psi_curr * np.exp(1j * phase_v)

        dcom_code = com_tgt_code - com_i_code
        if np.linalg.norm(dcom_code) > 0:
            Psi_curr = translate_op(Psi_curr, dcom_code)

    root_dir, fname = os.path.split(psi_path)
    stem = fname[:-4] if fname.endswith('.npy') else fname
    if stem.endswith('_psi'):
        initpath_root = os.path.join(root_dir, stem[:-4])
    else:
        initpath_root = os.path.join(root_dir, stem)
    deb_initpath = initpath_root + '_deboost'
    deb_psipath = deb_initpath + '_psi.npy'

    np.save(deb_psipath, Psi_curr.astype(np.complex128, copy=False))
    _add(f"[Targeting] Saved targeted file: {deb_psipath}")
    _add("           Use as InitPath:")
    _add(f"           InitPath = '{deb_initpath}'")

    Psi_final = np.load(deb_psipath)
    rho_f = np.abs(Psi_final) ** 2
    mask_f, _, _ = build_mask_around_COM(rho_f)
    M_f, Px_f, Py_f, Pz_f, v_f_tuple, COM_f_tuple = measure_MPv(Psi_final, mask_f)
    v_f_code = np.array([v_f_tuple[0], v_f_tuple[1], v_f_tuple[2]], dtype=float)
    v_f_kms = np.array(convert_between(v_f_code, '', 'km/s', 'v'), dtype=float)
    com_f_code = np.array([COM_f_tuple[0], COM_f_tuple[1], COM_f_tuple[2]], dtype=float)

    _add(f"  AFTER: v=[{v_f_kms[0]:.9e}, {v_f_kms[1]:.9e}, {v_f_kms[2]:.9e}] km/s; COM={com_f_code.tolist()} (code)")
    _add(
        f"  Errors: |Δv|={float(np.linalg.norm(v_f_kms - np.array(convert_between(v_tgt_code, '', 'km/s', 'v'), dtype=float))):.6e} km/s, "
        f"|ΔCOM|={float(np.linalg.norm(com_f_code - com_tgt_code)):.6e} (code)"
    )

    _final_flush()
    return {"psi_path": psi_path, "deb_initpath": deb_initpath, "deb_psipath": deb_psipath}


def compute_mode_fractions(
    *,
    eigs_data: Dict,
    r_grid: np.ndarray,
    length: float,
    length_units: str,
    resol: int,
    CellCenteredGrid: bool,
    nlm_list: Iterable[Tuple[int, int, int]],
    center_of_mass: Iterable[float] | None = None,
    epsilon_nlm_list: Iterable[complex] | None = None,
    only_nonzero_eps: bool = False,
    psi_deboost_path: str | None = None,
    deb_psipath: str | None = None,
    deb_initpath: str | None = None,
    init_basename: str | None = None,
    psi_path: str | None = None,
    nprocs_modes: int = 8,
):
    """Compute per-mode mass fractions for a deboosted Psi file."""
    from .units import convert_between

    sph_harm_ylm, ylm_backend = _get_sph_harm_backend()

    if psi_deboost_path is None:
        candidates: List[str] = []
        if deb_psipath is not None:
            candidates.append(str(deb_psipath))
        if deb_initpath is not None:
            candidates.append(str(deb_initpath) + '_psi.npy')
        if init_basename is not None:
            base = str(init_basename)
            if base.endswith('.npy'):
                base = base[:-4]
            candidates.extend([base + '_deboost_psi.npy', base + '_deboost.npy'])
        if psi_path is not None:
            base_dir = os.path.dirname(str(psi_path))
            for c in list(candidates):
                if c and not os.path.isabs(c):
                    candidates.append(os.path.join(base_dir, c))
        candidates = [os.path.abspath(c) if not os.path.isabs(c) else c for c in candidates if c]

        seen = set()
        uniq_candidates = []
        for c in candidates:
            if c not in seen:
                uniq_candidates.append(c)
                seen.add(c)
        for c in uniq_candidates:
            if os.path.exists(c):
                psi_deboost_path = os.path.abspath(c)
                break
        if psi_deboost_path is None:
            raise FileNotFoundError("未找到 deboost Psi 文件，尝试过：\n  " + "\n  ".join(uniq_candidates))

    print(f"[Mode fractions] Using Psi file: {psi_deboost_path}")
    print(f"[Mode fractions] Y_lm backend: {ylm_backend}")

    lengthC = float(convert_between(length, length_units, '', 'l'))
    N = int(resol)
    dx = lengthC / N
    if CellCenteredGrid:
        gridvec = -lengthC / 2.0 + dx / 2.0 + np.arange(N) * dx
    else:
        gridvec = np.linspace(-lengthC / 2.0, lengthC / 2.0, N, endpoint=False)

    cm_target_len = list(center_of_mass) if center_of_mass is not None else [0.0, 0.0, 0.0]
    com_code = np.array(convert_between(np.array(cm_target_len, dtype=float), length_units, '', 'l'), dtype=float)

    X, Y = np.meshgrid(gridvec, gridvec, indexing='ij')
    XR = X - com_code[0]
    YR = Y - com_code[1]
    phi2d = np.mod(np.arctan2(YR, XR), 2.0 * np.pi)
    r_xy2 = XR ** 2 + YR ** 2

    r_grid_arr = np.asarray(r_grid, dtype=float)
    unique_nl = []
    for n, l, m in nlm_list:
        key = (int(l), int(n))
        if key not in unique_nl:
            unique_nl.append(key)

    radial_fn = {}
    radial_norms = {}
    for l, n in unique_nl:
        key = f'ell/{int(l)}'
        if key not in eigs_data:
            raise KeyError(f"eigs_data does not contain {key}")
        f_mat = np.asarray(eigs_data[key]['f'])
        if n < 0 or n >= f_mat.shape[1]:
            raise IndexError(f"l={l} n={n} out of range (0..{f_mat.shape[1]-1})")
        f_vals = np.asarray(f_mat[:, int(n)], dtype=float)
        radial_fn[(l, n)] = (r_grid_arr, f_vals)
        radial_norm = float(np.trapz(np.abs(f_vals) ** 2 * (r_grid_arr ** 2), r_grid_arr))
        if radial_norm <= 0:
            raise ValueError(f"l={l}, n={n} radial norm is invalid: {radial_norm}")
        radial_norms[(l, n)] = radial_norm

    modes = [(int(n), int(l), int(m)) for (n, l, m) in nlm_list]
    if only_nonzero_eps and epsilon_nlm_list is not None:
        eps_arr = np.asarray(list(epsilon_nlm_list))
        modes = [m for i, m in enumerate(modes) if i < len(eps_arr) and eps_arr[i] != 0]
    modes = sorted(modes, key=lambda x: (x[1], x[0], x[2]))

    psi_mmap = np.load(psi_deboost_path, mmap_mode='r')
    if psi_mmap.shape != (N, N, N):
        raise ValueError(f"Psi shape {psi_mmap.shape} does not match {(N, N, N)}")

    total_mass = 0.0
    for k in range(N):
        total_mass += float(np.sum(np.abs(psi_mmap[:, :, k]) ** 2))
    total_mass *= dx ** 3
    print(f"[Mode fractions] Total mass from |Psi|^2 = {total_mass:.6e} (code units)")

    sum_mode_mass = 0.0
    print("\n[Mode fractions] Per-mode results:")
    for (n, l, m) in modes:
        rg, fr = radial_fn[(l, n)]
        norm = radial_norms[(l, n)]
        total = 0.0 + 0.0j
        with mp.get_context('fork').Pool(
            processes=nprocs_modes,
            initializer=_init_mode_fraction_worker,
            initargs=(
                psi_deboost_path,
                rg,
                fr,
                l,
                m,
                gridvec,
                com_code,
                phi2d,
                r_xy2,
                sph_harm_ylm,
            ),
        ) as pool:
            iterator = pool.imap(_mode_overlap_slice_worker, range(N), chunksize=1)
            if tqdm is not None:
                iterator = tqdm(
                    iterator,
                    total=N,
                    desc=f"Mode (n={n}, l={l}, m={m})",
                    unit="slice",
                    leave=False,
                )
            for val in iterator:
                total += val
        overlap = total * (dx ** 3)
        mode_mass = (np.abs(overlap) ** 2) / norm
        sum_mode_mass += mode_mass
        frac_pct = 100.0 * mode_mass / total_mass if total_mass > 0 else np.nan
        print(f"  (n={n}, l={l}, m={m}) mass={mode_mass:.6e}, frac={frac_pct:.6e}%")

    sum_frac_pct = 100.0 * sum_mode_mass / total_mass if total_mass > 0 else np.nan
    print(f"[Mode fractions] Sum of listed mode masses = {sum_mode_mass:.6e} (fraction={sum_frac_pct:.6e}%)")
    print(f"[Mode fractions] Unaccounted mass = {total_mass - sum_mode_mass:.6e} (code units)")

    return {
        "psi_deboost_path": psi_deboost_path,
        "total_mass": total_mass,
        "sum_mode_mass": sum_mode_mass,
        "sum_frac_pct": sum_frac_pct,
        "ylm_backend": ylm_backend,
    }