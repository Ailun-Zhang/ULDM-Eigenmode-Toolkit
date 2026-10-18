"""T9/T10/T11/T14 in c_nlm_integrator."""
import h5py
import numpy as np
import pytest
import scipy.interpolate as si

from Eig_ULDM_packages import c_nlm_integrator as cni
from Eig_ULDM_packages._h5meta import write_attrs


@pytest.mark.parametrize("name,expected", [
    ("PyUL_psi_alm_0.02_mass170_256.h5", None),          # the paper's file: 0.02 is the BH mass ratio
    ("PyUL_psi_alm_0.05_(1,0,0)_mass50_256.h5", None),
    ("PyUL_psi_alm.h5", None),
    ("PyUL_psi_alm_rmax0.206_mass170.h5", 0.206),
    ("x_rmax_0.20604798941694438_y.h5", 0.20604798941694438),
    ("x_r_max=1e-1.h5", 0.1),
    ("/some/dir/a_RMAX2.5.hdf5", 2.5),
])
def test_parse_rmax(name, expected):
    assert cni._parse_rmax_from_filename(name) == expected


def _write_alm(path, n_t=5, n_r=6, l_max=1, attrs=None):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f["massArr"] = np.ones(n_t)
        for i in range(n_t):
            g = f.create_group(f"alm_{i:06d}")
            g["counts"] = np.full((n_r, (l_max + 1) ** 2), 8.0)
            g["ylm"] = rng.normal(size=(n_r, (l_max + 1) ** 2)) + 1j
        if attrs:
            write_attrs(f.attrs, attrs)


def test_load_a_lm_time_axis(tmp_path):
    p = str(tmp_path / "a.h5")
    _write_alm(p)
    r = np.linspace(0.1, 1, 6)
    legacy = cni.load_a_lm(p, r_al=r, final_time=40.0)
    assert np.array_equal(legacy["t"], np.linspace(0, 40.0, 5)) and legacy["snapshot_indices"] is None
    sub = cni.load_a_lm(p, r_al=r, final_time=40.0, start_index=2, end_index=4)
    assert np.array_equal(sub["t"], np.linspace(0, 40.0, 3))   # historical: always starts at 0
    real = cni.load_a_lm(p, r_al=r, start_index=2, end_index=4, snapshot_indices=[502, 503, 504], save_interval=6.0)
    assert np.array_equal(real["t"], np.array([502, 503, 504]) * 6.0)
    expl = cni.load_a_lm(p, r_al=r, times=[1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.array_equal(expl["t"], [1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        cni.load_a_lm(p, r_al=r, time_source="file")


def test_load_a_lm_reads_attrs(tmp_path):
    p = str(tmp_path / "b.h5")
    times = np.array([0.0, 6.0, 12.0, 18.0, 24.0]) + 3000.0
    r_code = np.linspace(0.01, 0.2, 6)
    _write_alm(p, attrs={"times": times, "snapshot_indices": np.arange(500, 505), "r_values": r_code * 12133,
                         "r_units": "pc", "r_values_code": r_code})
    d = cni.load_a_lm(p, r_al=None, final_time=99.0, start_index=1, end_index=3)
    assert np.array_equal(d["t"], times[1:4]) and list(d["snapshot_indices"]) == [501, 502, 503]
    assert np.array_equal(d["r"], r_code)
    leg = cni.load_a_lm(p, r_al=None, final_time=99.0, time_source="legacy")
    assert np.array_equal(leg["t"], np.linspace(0, 99.0, 5))
    # r in physical units only -> clear error instead of a wrong grid
    p2 = str(tmp_path / "c.h5")
    _write_alm(p2, attrs={"r_values": r_code * 12133, "r_units": "pc"})
    with pytest.raises(ValueError, match="code units"):
        cni.load_a_lm(p2, r_al=None)


def test_check_rmax_consistency():
    assert cni.check_rmax_consistency(np.array([0.1, 0.2]), 0.2) == 0.0
    with pytest.raises(ValueError):
        cni.check_rmax_consistency(0.21, 0.2)
    with pytest.warns(RuntimeWarning):
        cni.check_rmax_consistency(0.21, 0.2, raise_error=False)
    r = np.linspace(0, 0.2, 11)
    r[0] = 1e-6
    assert np.array_equal(cni.rescale_r_to_eigen_rmax(r, 0.2), (r / r[-1]) * 0.2)


def test_nearest_is_pure_nearest():
    rng = np.random.default_rng(1)
    t_in = np.linspace(0, 10, 11)
    y = rng.normal(size=11) + 1j * rng.normal(size=11)
    t_out = np.linspace(0, 10, 101)
    pure = si.interp1d(t_in, y, kind="nearest", bounds_error=False, fill_value=(y[0], y[-1]), assume_sorted=True)(t_out)
    a = cni.resample_cnlm_time({(0, 0, 0): y}, t_in, t_out, method="nearest")[(0, 0, 0)]
    b = cni.resample_cnlm_time({(0, 0, 0): y}, t_in, t_out, method="nearest-savgol")[(0, 0, 0)]
    assert np.array_equal(a, pure)
    assert not np.allclose(b, pure)
    yr = rng.normal(size=11)
    ar = cni.resample_cnlm_time({(0, 0, 0): yr}, t_in, t_out, method="nearest")[(0, 0, 0)]
    assert np.array_equal(ar, si.interp1d(t_in, yr, kind="nearest", bounds_error=False,
                                          fill_value=(yr[0], yr[-1]), assume_sorted=True)(t_out))


def test_compute_cnlm_parallel_equals_serial():
    rng = np.random.default_rng(2)
    r_f = np.linspace(0, 1, 200)
    r_al = np.linspace(1e-3, 1, 30)
    f_dict = {(l, n): (np.sin((n + 1) * np.pi * r_f) * r_f ** l).astype(np.complex128) for l in range(3) for n in range(3)}
    a_lm = {(l, m): rng.normal(size=(30, 7)) + 1j * rng.normal(size=(30, 7)) for l in range(3) for m in range(-l, l + 1)}
    t = np.linspace(0, 1, 7)
    _, c1 = cni.compute_cnlm(r_f, f_dict, r_al, t, a_lm, num_workers=1)
    _, c2 = cni.compute_cnlm(r_f, f_dict, r_al, t, a_lm, num_workers=2)
    assert set(c1) == set(c2) and len(c1) == 27
    for k in c1:
        assert np.array_equal(c1[k], c2[k])


def test_save_cnlm_metadata_roundtrip(tmp_path):
    p = str(tmp_path / "c.h5")
    cni.save_cnlm_to_h5({(0, 1, -1): np.array([1j, 2.0])}, np.array([0.0, 1.0]), p,
                        metadata={"source": "x", "params": {"a": 1}, "arr": np.arange(3), "none": None,
                                  "big": np.arange(10000.0)})
    t, c, attrs = cni.load_cnlm_from_h5(p)
    assert attrs["source"] == "x" and '"a": 1' in attrs["params"] and list(attrs["arr"]) == [0, 1, 2]
    with h5py.File(p) as f:
        assert sorted(f.keys()) == ["n", "t"]
        from Eig_ULDM_packages._h5meta import read_array_attr
        assert np.array_equal(read_array_attr(f.attrs, "big"), np.arange(10000.0))
