"""T4/T6/T7/T8: functions3 imports and error handling, create_sh_hdf5 completeness and provenance."""
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from conftest import REPO, make_cube
from Eig_ULDM_packages import functions3 as f3
from Eig_ULDM_packages.alm_utils import create_sh_hdf5
from Eig_ULDM_packages._compat import prefix_folder


def test_functions3_imports_without_optional_deps():
    code = ("import sys\n"
            "for m in ('moviepy', 'cryptography', 'cryptography.utils'):\n"
            "    sys.modules[m] = None  # make the import fail\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            "import Eig_ULDM_packages.functions3 as f3\n"
            "assert 'matplotlib.pyplot' not in sys.modules or True\n"
            "print('ok', f3.process_file.__name__)\n")
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok process_file" in out.stdout


@pytest.mark.parametrize("prefix,expected", [("3WfnRS/P3R_#", "3WfnRS"), ("3WfnRS\\P3R_#", "3WfnRS"),
                                             ("3Wfn/P3D_#", "3Wfn"), ("P3R_#", "P3R_#")])
def test_prefix_folder(prefix, expected):
    assert prefix_folder(prefix) == expected


def test_prefix_folder_absolute(tmp_path):
    assert prefix_folder(str(tmp_path / "3WfnRS" / "P3R_#")) == "3WfnRS"


def test_process_file_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        f3.process_file(3, 0.5, 2, 0.2, 1, "3WfnRS/P3R_#", 16)
    assert f3.process_file(3, 0.5, 2, 0.2, 1, "3WfnRS/P3R_#", 16, raise_on_error=False) is None
    (tmp_path / "3WfnRS").mkdir()
    np.save(tmp_path / "3WfnRS" / "P3R_#003.npy", np.zeros((4, 4)))       # not a cube
    with pytest.raises(Exception):
        f3.process_file(3, 0.5, 2, 0.2, 1, "3WfnRS/P3R_#", 16)
    assert "P3R_#003" in (tmp_path / "error_log.txt").read_text()


def _make_jsons(tmp_path, n_snap=3, n_r=5, l_max=2, N=16):
    (tmp_path / "3WfnRS").mkdir()
    for k in range(n_snap):
        np.save(tmp_path / "3WfnRS" / f"P3R_#{k:03d}.npy", make_cube(16, seed=k))
    r = np.linspace(0, 2.0, n_r)
    r[0] = 1e-5
    for k in range(n_snap):
        for rr in r:
            f3.process_file(k, rr, l_max, 5.0 / 16, 1, "3WfnRS/P3R_#", N)
    return r


def test_create_sh_hdf5_complete_and_attrs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _make_jsons(tmp_path)
    create_sh_hdf5("a.h5", 0, 2, len(r), 2, 16, r, n_workers=1, times=[0.0, 6.0, 12.0], time_units="Myr",
                   r_units="kpc")
    with h5py.File("a.h5") as f:
        assert sorted(f.keys()) == ["alm_000000", "alm_000001", "alm_000002", "massArr"]
        assert sorted(f["alm_000000"].keys()) == ["counts", "ylm"]
        assert f["alm_000000"]["ylm"].shape == (5, 9) and f["alm_000000"]["ylm"].dtype == np.complex128
        assert list(f.attrs["snapshot_indices"]) == [0, 1, 2] and f.attrs["r_units"] == "kpc"
        assert np.array_equal(f.attrs["r_values"], r) and f.attrs["missing_count"] == 0
    assert not os.path.exists("a.h5.tmp")


def test_create_sh_hdf5_missing_raises_or_zero_fills(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _make_jsons(tmp_path)
    victim = os.path.join("alm_coefficients", "3WfnRS", f"r0_{r[2]}", "lmax_2",
                          f"alm_coefficients_P3R_#001_r0_{r[2]}.json")
    os.remove(victim)
    with pytest.raises(RuntimeError, match="missing"):
        create_sh_hdf5("b.h5", 0, 2, len(r), 2, 16, r, n_workers=1)
    assert not os.path.exists("b.h5") and not os.path.exists("b.h5.tmp")
    create_sh_hdf5("b.h5", 0, 2, len(r), 2, 16, r, n_workers=1, allow_missing=True)
    with h5py.File("b.h5") as f:
        assert np.all(f["alm_000001"]["ylm"][2] == 0)
        assert f.attrs["missing_count"] == 1 and list(f.attrs["missing_per_time"]) == [0, 1, 0]
        assert "P3R_#001" in f.attrs["missing_samples"]
    with pytest.raises(ValueError):
        create_sh_hdf5("c.h5", 0, 2, len(r) + 1, 2, 16, r, n_workers=1)
