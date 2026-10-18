"""T17: 'legacy' (historical) vs 'dh' (Driscoll-Healy) sphere sampling on analytic Y_lm."""
import numpy as np
import pyshtools as pysh
import pytest

from Eig_ULDM_packages.alm_kernel import angular_grid_1d, flatten_shtools_coeffs, lm_list
from Eig_ULDM_packages._compat import sph_harm_ylm


def expand_ylm(l, m, N, grid, l_max):
    th, ph = angular_grid_1d(N, grid)
    T, P = np.meshgrid(th, ph, indexing="ij")
    f = pysh.expand.spharm_lm(l, m, T, P, normalization="ortho", kind="complex", csphase=-1, degrees=False)
    raw = pysh.expand.SHExpandDHC(np.ascontiguousarray(f), norm=4, sampling=2, csphase=-1, lmax_calc=l_max)
    return flatten_shtools_coeffs(raw, l_max)


@pytest.mark.parametrize("l,m", [(0, 0), (1, 0), (1, 1), (2, -2), (5, 3), (10, 10)])
def test_dh_grid_is_exact(l, m):
    c = expand_ylm(l, m, 64, "dh", 12)
    target = np.zeros_like(c)
    target[lm_list(12).index((l, m))] = 1.0
    assert np.max(np.abs(c - target)) < 1e-10


def test_legacy_grid_bias_is_documented():
    # Y00 exact; l > 0 picks up a constant amplitude/phase error and leakage of order l/N (paper default N=300).
    c00 = expand_ylm(0, 0, 300, "legacy", 10)
    assert abs(c00[0] - 1) < 1e-12 and np.max(np.abs(c00[1:])) < 1e-12
    c = expand_ylm(1, 1, 300, "legacy", 10)
    i = lm_list(10).index((1, 1))
    assert 1e-4 < abs(abs(c[i]) - 1) < 1e-2
    assert 1e-3 < abs(np.angle(c[i])) < 1e-2          # ~ m*pi/(2N-1)
    leak = np.max(np.abs(np.delete(c, i)))
    assert 1e-4 < leak < 3e-2


@pytest.mark.parametrize("l,m", [(0, 0), (1, -1), (2, 1), (4, -3), (6, 6)])
def test_scipy_ylm_wrapper_matches_pyshtools(l, m):
    rng = np.random.default_rng(l * 10 + m)
    th = rng.uniform(0, np.pi, 50)
    ph = rng.uniform(0, 2 * np.pi, 50)
    a = sph_harm_ylm(l, m, th, ph)
    b = pysh.expand.spharm_lm(l, m, th, ph, normalization="ortho", kind="complex", csphase=-1, degrees=False)
    assert np.allclose(a, b, rtol=1e-10, atol=1e-12)
