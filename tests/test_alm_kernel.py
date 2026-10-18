"""K1: one-pass a_lm kernel vs the historical functions3.process_file (bitwise)."""
import numpy as np
import pytest
from scipy import ndimage

from conftest import make_cube
from Eig_ULDM_packages import functions3 as f3
from Eig_ULDM_packages.alm_kernel import (AlmKernel, angular_grid_1d, compute_alm_cube, flatten_shtools_coeffs,
                                          lm_list, mass_center_index_space, prefilter_cube)
from Eig_ULDM_packages.alm_utils import reorganize_coeffs


@pytest.fixture(scope="module")
def cube():
    return make_cube(24, seed=3)


def test_com_bitwise(cube):
    dx = 5.0 / 24
    assert np.array_equal(f3.calculate_mass_center(cube, dx), mass_center_index_space(cube, dx))


def test_prefilter_once_bitwise(cube):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-4, 28, size=(3, 20000))  # includes points outside the cube (cval=0 region)
    re, im = prefilter_cube(cube)
    for comp, coef in ((cube.real, re), (cube.imag, im)):
        a = ndimage.map_coordinates(comp, pts, order=3, mode="constant", cval=0.0)
        b = ndimage.map_coordinates(coef, pts, order=3, mode="constant", cval=0.0, prefilter=False)
        assert np.array_equal(a, b)


def test_angular_grids_match_functions3():
    for g in ("legacy", "dh"):
        a = angular_grid_1d(40, g)
        b = f3.angular_grid_1d(40, g)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    th, ph = angular_grid_1d(40, "legacy")
    assert np.array_equal(th, np.linspace(0, np.pi, 40)) and np.array_equal(ph, np.linspace(0, 2 * np.pi, 80))
    th, ph = angular_grid_1d(40, "dh")
    assert th[0] == 0 and th[-1] < np.pi and ph[-1] < 2 * np.pi


@pytest.mark.parametrize("workers,chunk", [(1, None), (4, None), (3, 1000), (64, None)])
def test_kernel_bitwise_vs_process_file(cube, tmp_path, monkeypatch, workers, chunk):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "3WfnRS").mkdir()
    np.save(tmp_path / "3WfnRS" / "P3R_#007.npy", cube)
    dx, l_max, N = 5.0 / 24, 4, 40
    r_values = np.linspace(0, 2.5, 13)
    r_values[0] = 1e-5
    for r in r_values:
        out = f3.process_file(7, r, l_max, dx, 1, "3WfnRS/P3R_#", N)
        assert out.endswith(".json")
    raw_json = np.array([f3.load_alm_from_file("3WfnRS/P3R_#", 7, r, l_max)[1] for r in r_values])
    raw = compute_alm_cube(cube, dx, r_values, l_max, num_points=N, workers=workers, chunk_points=chunk,
                           return_raw=True)
    assert np.array_equal(raw, raw_json)
    flat = compute_alm_cube(cube, dx, r_values, l_max, num_points=N, workers=workers, chunk_points=chunk)
    assert np.array_equal(flat, np.array([reorganize_coeffs(c, l_max) for c in raw_json]))
    assert np.array_equal(flatten_shtools_coeffs(raw, l_max), flat)


@pytest.mark.parametrize("dtype", [np.complex64, np.float32])
def test_kernel_bitwise_vs_process_file_single_precision(cube, tmp_path, monkeypatch, dtype):
    """R9-04: single-precision cubes are sampled in float32, exactly as process_file does."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "3WfnRS").mkdir()
    c32 = (cube if np.dtype(dtype).kind == "c" else cube.real).astype(dtype)
    np.save(tmp_path / "3WfnRS" / "P3R_#002.npy", c32)
    dx, l_max, N = 5.0 / 24, 4, 16
    r_values = np.linspace(0, 2.5, 9)
    r_values[0] = 1e-5
    for r in r_values:
        f3.process_file(2, r, l_max, dx, 1, "3WfnRS/P3R_#", N)
    raw_json = np.array([f3.load_alm_from_file("3WfnRS/P3R_#", 2, r, l_max)[1] for r in r_values])
    for workers in (1, 4):
        raw = compute_alm_cube(c32, dx, r_values, l_max, num_points=N, workers=workers, return_raw=True)
        assert np.array_equal(raw, raw_json)
    # and the float64 sampling of the same cube is measurably different (so the test has teeth)
    raw64 = compute_alm_cube(c32.astype(np.complex128), dx, r_values, l_max, num_points=N, workers=1,
                             return_raw=True)
    assert not np.array_equal(raw64, raw_json)


def test_dh_grid_option_matches_process_file(cube, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "3WfnRS").mkdir()
    np.save(tmp_path / "3WfnRS" / "P3R_#000.npy", cube)
    r = np.array([0.4, 0.9])
    for rr in r:
        f3.process_file(0, rr, 3, 5.0 / 24, 1, "3WfnRS/P3R_#", 20, angular_grid="dh")
    raw_json = np.array([f3.load_alm_from_file("3WfnRS/P3R_#", 0, rr, 3)[1] for rr in r])
    raw = compute_alm_cube(cube, 5.0 / 24, r, 3, num_points=20, angular_grid="dh", return_raw=True, workers=2)
    assert np.array_equal(raw, raw_json)


def test_explicit_center_and_return_center(cube):
    dx = 5.0 / 24
    k = AlmKernel([0.5, 1.0], 2, num_points=16, workers=2)
    c = mass_center_index_space(cube, dx)
    a1, c1 = k.compute(cube, dx, center="com", return_center=True)
    a2 = k.compute(cube, dx, center=c)
    assert np.array_equal(c1, c) and np.array_equal(a1, a2)
    a3 = k.compute(cube, dx, center=c + np.array([dx, 0, 0]))
    assert not np.array_equal(a1, a3)


def test_validation(cube):
    with pytest.raises(ValueError):
        AlmKernel([1.0], 2, num_points=15)          # odd N
    with pytest.raises(ValueError):
        AlmKernel([1.0], 20, num_points=20)         # l_max > N/2-1
    with pytest.raises(ValueError):
        AlmKernel([1.0], 2, num_points=16, angular_grid="gauss")
    k = AlmKernel([1.0], 2, num_points=16, workers=1)
    with pytest.raises(ValueError):
        k.compute(cube[:, :, :5], 0.1)
    with pytest.raises(ValueError):
        k.compute(cube, -1.0)
    with pytest.raises(ValueError):
        k.compute(np.zeros_like(cube), 0.1)         # COM of an empty cube is NaN


def test_lm_list_order():
    assert lm_list(1) == [(0, 0), (1, -1), (1, 0), (1, 1)]
