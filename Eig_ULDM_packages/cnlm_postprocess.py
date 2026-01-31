"""Helpers for preparing epsilon_nlm from c_nlm outputs."""

from __future__ import annotations

from typing import Iterable, Tuple, Dict, List

import numpy as np


def _format_nlm_label(n: int, l: int, m: int) -> str:
    return f"{n}{l}{m}"


def prepare_epsilon_nlm(
    nlm_list: Iterable[Tuple[int, int, int]],
    *,
    use_file: bool = True,
    c_file: Dict[Tuple[int, int, int], np.ndarray] | None = None,
    t_point: int | None = None,
    mass: float | None = None,
    epsilon_nlm_list: Iterable[complex] | None = None,
    normalize_c000: bool = True,
    print_coeffs: bool = True,
    print_norm: bool = True,
    inject_globals: dict | None = None,
):
    """Prepare epsilon_nlm_list from c_nlm data or manual input.

    Parameters
    ----------
    nlm_list : iterable of (n,l,m)
        Mode list order used for epsilon output.
    use_file : bool
        If True, read from c_file; if False, use epsilon_nlm_list directly.
    c_file : dict
        Mapping (n,l,m) -> complex time series.
    t_point : int
        Time index to sample from c_file.
    mass : float
        Mass used to rescale c_nlm values.
    epsilon_nlm_list : iterable of complex
        Manual epsilon list (used when use_file=False).
    normalize_c000 : bool
        If True and (0,0,0) is present, normalize c000 to enforce sum |c|^2 = 1.
    print_coeffs : bool
        If True, print c_{nlm} values.
    print_norm : bool
        If True, print |c|^2 values and normalization check.
    inject_globals : dict
        If provided, inject per-mode coefficients into this namespace
        (e.g. globals()), and also c000_unnorm / c000_norm when available.
    """
    nlm_list = list(nlm_list)
    coeffs: Dict[Tuple[int, int, int], complex] = {}
    c000_unnorm = None
    c000_norm = None

    if use_file:
        if c_file is None or t_point is None or mass is None:
            raise ValueError("c_file, t_point, and mass are required when use_file=True.")
        for (n, l, m) in nlm_list:
            coeffs[(n, l, m)] = c_file[(n, l, m)][t_point] / np.sqrt(mass)

        if normalize_c000 and (0, 0, 0) in coeffs:
            c000_unnorm = coeffs[(0, 0, 0)]
            other_sum = 0.0
            for key, val in coeffs.items():
                if key != (0, 0, 0):
                    other_sum += np.abs(val) ** 2
            c000_norm = np.sqrt(1 - other_sum)
            coeffs[(0, 0, 0)] = c000_unnorm / np.abs(c000_unnorm) * c000_norm
    else:
        if epsilon_nlm_list is None:
            raise ValueError("epsilon_nlm_list is required when use_file=False.")
        eps_list = list(epsilon_nlm_list)
        if len(eps_list) != len(nlm_list):
            raise ValueError("epsilon_nlm_list length must match nlm_list length.")
        for idx, key in enumerate(nlm_list):
            coeffs[key] = eps_list[idx]

    if print_coeffs:
        for (n, l, m) in nlm_list:
            label = _format_nlm_label(n, l, m)
            print(f"c_{{{label}}}={coeffs[(n, l, m)]}")

    check = None
    if print_norm:
        for (n, l, m) in nlm_list:
            label = _format_nlm_label(n, l, m)
            print(f"|c_{{{label}}}|^2={np.abs(coeffs[(n, l, m)])**2}")
        check = sum(np.abs(coeffs[key]) ** 2 for key in nlm_list)
        print(f"Check normalization: {check}")

    eps_out: List[complex] = [coeffs[key] for key in nlm_list]

    if inject_globals is not None:
        for (n, l, m), val in coeffs.items():
            var_name = f"c{n}{l}{m}".replace("-", "_")
            inject_globals[var_name] = val
        if c000_unnorm is not None:
            inject_globals["c000_unnorm"] = c000_unnorm
        if c000_norm is not None:
            inject_globals["c000_norm"] = c000_norm
        if check is not None:
            inject_globals["check"] = check

    return eps_out, coeffs, check