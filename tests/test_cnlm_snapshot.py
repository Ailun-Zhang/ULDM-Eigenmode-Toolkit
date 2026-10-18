"""K2: per-snapshot projector and writers vs the historical chain (same process => bitwise)."""
import h5py
import numpy as np
import pytest

from conftest import make_cube
from _legacy_chain import assert_h5_bitwise, h5_datasets, run_legacy_chain
from Eig_ULDM_packages.alm_kernel import compute_alm_cube
from Eig_ULDM_packages.c_nlm_integrator import load_cnlm_from_h5, load_eigenfunctions
from Eig_ULDM_packages.cnlm_snapshot import (CnlmProjector, alm_counts_roundtrip, cnlm_from_alm_snapshot,
                                             legacy_alm_radial_grid, stack_cnlm_snapshots, write_alm_h5,
                                             write_cnlm_h5)


def test_counts_roundtrip_is_not_identity_but_matches_h5(tmp_path):
    rng = np.random.default_rng(5)
    a = (rng.normal(size=(30, 9)) + 1j * rng.normal(size=(30, 9))) * 10.0 ** rng.uniform(-9, 1, size=(30, 9))
    rt = alm_counts_roundtrip(a, 300)
    assert not np.array_equal(rt, a)          # last-bit changes are real
    assert np.allclose(rt, a, rtol=1e-15, atol=0)
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f["ylm"] = a * (300 * 600)
        f["counts"] = np.full(a.shape, 300 * 600, dtype=np.float64)
    with h5py.File(p) as f:
        back = np.where(f["counts"][()] != 0, f["ylm"][()] / f["counts"][()], 0.0)
    assert np.array_equal(back, rt)


def test_snapshot_chain_bitwise(tmp_path, small_eigen_file):
    snap = tmp_path / "snaps"
    snap.mkdir()
    cubes = [make_cube(24, seed=s) for s in range(3)]
    for k, c in enumerate(cubes):
        np.save(snap / f"P3R_#{k:03d}.npy", c)
    dx, r_max, n_r, l_max, N = 5.0 / 24, 2.5, 13, 2, 24
    leg = run_legacy_chain(tmp_path / "legacy", snap, [0, 1, 2], dx, r_max, n_r, l_max, N, small_eigen_file,
                           final_time=20.0, cnlm_workers=1)
    r_kpc = np.linspace(0, r_max, n_r)
    r_kpc[0] = 1e-5
    alms = [compute_alm_cube(c, dx, r_kpc, l_max, num_points=N, workers=3) for c in cubes]
    r_f, f_dict = load_eigenfunctions(small_eigen_file)
    r_al = legacy_alm_radial_grid(r_max, n_r, float(r_f[-1]))
    assert np.array_equal(r_al, leg["r_al"])
    proj = CnlmProjector(r_f, f_dict, r_al, l_max)
    snaps = [proj.project(alm_counts_roundtrip(a, N)) for a in alms]
    c_dict = stack_cnlm_snapshots(snaps)
    assert list(c_dict.keys()) == list(leg["c_dict"].keys())        # serial compute_cnlm order
    for k in c_dict:
        assert np.array_equal(c_dict[k], leg["c_dict"][k]), k
    # one-shot helper
    one = cnlm_from_alm_snapshot(alms[1], r_al, r_f, f_dict, num_points_roundtrip=N)
    assert all(one[k] == c_dict[k][1] for k in one)
    # writers: identical datasets to the historical files
    write_cnlm_h5(str(tmp_path / "c_new.h5"), np.linspace(0.0, 20.0, 3), c_dict, metadata={"producer": "test"})
    assert_h5_bitwise(leg["cnlm"], tmp_path / "c_new.h5")
    write_alm_h5(str(tmp_path / "alm_new.h5"), alms, N, metadata={"r_values": r_kpc, "nested": {"a": 1}})
    assert_h5_bitwise(leg["alm"], tmp_path / "alm_new.h5")
    t, c_back, attrs = load_cnlm_from_h5(str(tmp_path / "c_new.h5"))
    assert attrs["producer"] == "test" and np.array_equal(t, np.linspace(0.0, 20.0, 3))


def test_writers_refuse_overwrite(tmp_path):
    p = str(tmp_path / "c.h5")
    write_cnlm_h5(p, np.arange(2.0), {(0, 0, 0): np.array([1 + 1j, 2 + 0j])})
    with pytest.raises(FileExistsError):
        write_cnlm_h5(p, np.arange(2.0), {(0, 0, 0): np.array([1 + 1j, 2 + 0j])})
    write_cnlm_h5(p, np.arange(2.0), {(0, 0, 0): np.array([3 + 1j, 2 + 0j])}, overwrite=True)
    assert load_cnlm_from_h5(p)[1][(0, 0, 0)][0] == 3 + 1j


def test_rmax_mismatch_raises(small_eigen_file):
    r_f, _ = load_eigenfunctions(small_eigen_file)
    with pytest.raises(ValueError, match="relative mismatch"):
        legacy_alm_radial_grid(2.4, 13, float(r_f[-1]))
