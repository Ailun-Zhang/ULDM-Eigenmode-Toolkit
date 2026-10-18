"""T1/T16: eigen solver (NumPy-2 trapezoid, outer boundary option, metadata, potential-from-grid)."""
import warnings

import h5py
import numpy as np
import pytest
import scipy.sparse.linalg as spla
from scipy.special import spherical_jn
from scipy.optimize import brentq

from Eig_ULDM_packages import uldm_eig1 as ue


def _bessel_zero(l, n):
    """n-th (0-based) positive zero of the spherical Bessel function j_l."""
    zs, x, f0 = [], 0.5, spherical_jn(l, 0.5)
    while len(zs) <= n:
        x1 = x + 0.05
        f1 = spherical_jn(l, x1)
        if f0 * f1 < 0:
            zs.append(brentq(lambda t: spherical_jn(l, t), x, x1))
        x, f0 = x1, f1
    return zs[n]


def test_legacy_default_is_unchanged():
    # Reference: the historical algorithm (interior solve H[1:,1:], wall effectively at r_max + h)
    r = np.linspace(0, 1.0, 301)
    phi = -1.0 / np.sqrt(r ** 2 + 0.05)
    H, _ = ue.make_h_matrix(r, phi, 1)
    E_ref = np.sort(spla.eigsh(H[1:, 1:], k=4, sigma=float(phi.min()), which="LM", tol=1e-12)[0])
    E, f = ue.solve_h(H, r, 4, shift=float(phi.min()), tol=1e-12, ell=1)
    assert np.allclose(E, E_ref, rtol=1e-11, atol=1e-11)
    E2, f2 = ue.solve_h(H, r, 4, shift=float(phi.min()), tol=1e-12, ell=1, outer_boundary="legacy")
    assert np.array_equal(E, E2) or np.allclose(E, E2, rtol=1e-12)
    assert f[-1, 0] != 0.0                      # the last point is an unknown in the legacy scheme


@pytest.mark.parametrize("ell", [0, 1, 2])
def test_outer_boundary_rmax_converges_to_fixed_domain(ell):
    # Free particle in a sphere of radius R: E_n = z_{l,n}^2 / (2 R^2) (hbar/m = 1)
    R = 1.0
    errs = {}
    for bc in ("legacy", "rmax"):
        for n_grid in (401, 801):
            r = np.linspace(0, R, n_grid)
            d = ue.make_all_eigs(r, np.zeros_like(r), ell, 3, shift=0.0, tol=1e-12, outer_boundary=bc)
            E = d[f"ell/{ell}"]["E"]
            exact = np.array([_bessel_zero(ell, n) ** 2 / (2 * R ** 2) for n in range(3)])
            errs[(bc, n_grid)] = np.max(np.abs(E / exact - 1))
            if bc == "rmax":
                assert np.all(d[f"ell/{ell}"]["f"][-1] == 0.0)
    # fixed domain: second-order convergence to the exact values
    assert errs[("rmax", 801)] < errs[("rmax", 401)] / 3 and errs[("rmax", 801)] < 1e-4
    # legacy: first-order bias from the wall at R + h (about 2h/R)
    assert errs[("legacy", 801)] > 5 * errs[("rmax", 801)]
    with pytest.raises(ValueError):
        ue.make_all_eigs(np.linspace(0, 1, 50), np.zeros(50), 0, 2, outer_boundary="bogus")


def test_compute_potential_and_save(tmp_path):
    import importlib.resources as resources
    sol = ue.load_soliton_file(str(resources.files("Eig_ULDM_packages") / "soliton.h5"))
    r = np.linspace(0, 0.2, 256)
    phi = ue.compute_potential(r, ue.soliton_density_function(170.0, sol))
    assert np.isfinite(phi).all() and phi[0] < phi[-1] < 0
    d = ue.make_all_eigs(r, phi, 1, 2)
    p = tmp_path / "e.h5"
    ue.save_eigs_to_h5(d, str(p), metadata={"soliton_mass_code": 170.0})
    with h5py.File(p) as f:
        assert sorted(f.keys()) == ["ell"] and sorted(f["ell"]["0"].keys()) == ["E", "f", "r"]
        assert f.attrs["soliton_mass_code"] == 170.0 and "version_numpy" in f.attrs


def test_soliton_mass0_override():
    import importlib.resources as resources
    path = str(resources.files("Eig_ULDM_packages") / "soliton.h5")
    r = np.linspace(0, 0.3, 30001)
    masses = {}
    for m0 in (None, 3.8827652755822006):
        sol = ue.load_soliton_file(path, mass0=m0)
        rho = ue.soliton_density_function(170.0, sol)(r)
        masses[m0] = float(np.sum(4 * np.pi * r ** 2 * rho) * (r[1] - r[0]))
    assert abs(masses[None] / 170.0 - 1) > 3e-5              # rounded mass0=3.883: ~6e-5 too light
    assert abs(masses[3.8827652755822006] / 170.0 - 1) < 1e-5


def test_potential_from_grid_conventions(tmp_path):
    N, L = 32, 2.0
    dx = L / N
    for cc in (True, False):
        ax = (np.arange(N) + (0.5 if cc else 0.0)) * dx - L / 2
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        pot = -(X ** 2 + Y ** 2 + Z ** 2)            # spherical about the origin: phi(r) = -r^2
        p = tmp_path / f"G_{cc}.npy"
        np.save(p, pot)
        r = np.linspace(0.05, 0.6, 12)
        line = ue.compute_potential_from_grid(r, str(p), dx=dx, method="line_cut", direction="x",
                                              cell_centered=cc)
        assert np.allclose(line, -r ** 2, atol=dx ** 2), cc
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ue.compute_potential_from_grid(np.linspace(0.05, 0.6, 12), str(tmp_path / "G_True.npy"), strict=False)
    assert any("inferred dx" in str(x.message) for x in w)
