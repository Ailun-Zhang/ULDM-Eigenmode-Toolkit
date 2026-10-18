"""K3: local pipeline CLI (synthetic PyUL run) vs the historical chain, resume, refusal, metadata."""
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from conftest import REPO, write_run
from _legacy_chain import assert_h5_bitwise, run_legacy_chain
from Eig_ULDM_packages import pipeline
from Eig_ULDM_packages.c_nlm_integrator import load_a_lm, load_cnlm_from_h5
from Eig_ULDM_packages.pyul_run import PyULRun, rs_geometry


def test_parse_snapshot_spec():
    assert pipeline.parse_snapshot_spec(None, 3) == [0, 1, 2, 3]
    assert pipeline.parse_snapshot_spec("1:3", 5) == [1, 2, 3]
    assert pipeline.parse_snapshot_spec("0:10:5,7", 10) == [0, 5, 7, 10]
    with pytest.raises(ValueError):
        pipeline.parse_snapshot_spec("0:11", 10)


def test_run_metadata(tmp_path):
    run = PyULRun.open(write_run(tmp_path / "a") / "Outputs")
    assert run.save_number == 3 and run.axion_mass_ev() == 1e-21 and run.duration_myr == 30.0
    assert list(run.snapshot_times_myr([0, 3])) == [0.0, 30.0]
    g = rs_geometry(run, cube_resol=24)
    assert g.side == 5.0 and g.resol == 24 and g.source == "reproducibility.uldm"
    old = PyULRun.open(write_run(tmp_path / "b", rs_record=False))
    with pytest.raises(ValueError, match="not recorded"):
        rs_geometry(old, cube_resol=24)
    assert rs_geometry(old, rs_side=5.0, rs_resol=24).dx == 5.0 / 24


def test_pipeline_equals_legacy_chain(tmp_path, small_eigen_file):
    run = write_run(tmp_path / "run", n=24, n_snap=4)
    out = tmp_path / "pipe"
    args = [str(run), "--eigen-file", small_eigen_file, "--num-points", "24", "--n-shells", "13",
            "--workers", "3", "--out-dir", str(out), "--alm-out", "alm.h5"]
    res = pipeline.run_pipeline(pipeline.build_parser().parse_args(args))
    leg = run_legacy_chain(tmp_path / "legacy", run / "Outputs" / "3WfnRS", [0, 1, 2, 3], 5.0 / 24, 2.5, 13, 2, 24,
                           small_eigen_file, final_time=30.0)
    assert_h5_bitwise(leg["alm"], res["alm_file"])
    assert_h5_bitwise(leg["cnlm"], res["cnlm_file"])
    t, c, attrs = load_cnlm_from_h5(res["cnlm_file"])
    assert attrs["axion_mass_ev"] == 1e-21 and attrs["angular_grid"] == "legacy"
    assert attrs["rs_source"] == "reproducibility.uldm" and list(attrs["snapshot_indices"]) == [0, 1, 2, 3]
    # the a_lm file is self-describing: load_a_lm needs no r_al / final_time
    d = load_a_lm(res["alm_file"], r_al=None)
    assert np.array_equal(d["r"], leg["r_al"]) and np.array_equal(d["t"], t)
    # resume: final outputs are protected, checkpoints are reused
    with pytest.raises(SystemExit, match="exists"):
        pipeline.run_pipeline(pipeline.build_parser().parse_args(args))
    os.remove(res["cnlm_file"])
    os.remove(res["alm_file"])
    res2 = pipeline.run_pipeline(pipeline.build_parser().parse_args(args))
    assert_h5_bitwise(leg["cnlm"], res2["cnlm_file"])
    # changed kernel parameters on the same out-dir are refused
    args2 = args[:-2] + ["--num-points", "20", "--cnlm-out", "c2.h5"]
    with pytest.raises(SystemExit, match="different kernel parameters"):
        pipeline.run_pipeline(pipeline.build_parser().parse_args(args2))
    # --force recomputes (and replaces the parameter record)
    res3 = pipeline.run_pipeline(pipeline.build_parser().parse_args(args2 + ["--force"]))
    assert os.path.exists(res3["cnlm_file"])
    summary = json.loads((out / "pipeline_summary.json").read_text())
    assert summary["l_max"] == 2 and summary["num_points"] == 20


def test_pipeline_cli_build_eigen_subset_and_errors(tmp_path):
    run = write_run(tmp_path / "run", n=16, n_snap=6, save_number=10, rs_record=False)
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    env.pop("PYUL_AXION_MASS_EV", None)
    base = [sys.executable, "-B", "-m", "Eig_ULDM_packages.pipeline", str(run)]
    # no RS geometry recorded (PyUL <= 30.40) -> explicit values required
    r = subprocess.run(base + ["--snapshots", "0:5", "--build-eigen", "--out-dir", str(tmp_path / "o1")], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "--rs-side" in (r.stdout + r.stderr)
    # snapshots 6..10 missing -> error unless --skip-missing / --snapshots
    r = subprocess.run(base + ["--rs-side", "5", "--build-eigen", "--out-dir", str(tmp_path / "o1")], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "missing" in (r.stdout + r.stderr)
    out = tmp_path / "o2"
    r = subprocess.run(base + ["--rs-side", "5", "--snapshots", "0:5:2", "--build-eigen", "--eigen-n-grid", "96",
                               "--eigen-lmax", "1", "--eigen-n-eig", "2", "--num-points", "16", "--workers", "2",
                               "--out-dir", str(out)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    t, c, attrs = load_cnlm_from_h5(str(out / "c_nlm.h5"))
    assert np.allclose(t, [0.0, 6.0, 12.0])                 # true save times of snapshots 0, 2, 4 (30 Myr / 10 saves)
    assert len(c) == 2 * 1 + 2 * 3 and attrs["rs_source"] == "explicit side"
    eig = [p for p in os.listdir(out) if p.startswith("eigs_mass_")][0]
    with h5py.File(out / eig) as f:
        assert f.attrs["n_grid"] == 96 and f.attrs["axion_mass_ev"] == 1e-21
    r = subprocess.run(base + ["--rs-side", "5", "--snapshots", "0:5", "--dry-run", "--eigen-file", str(out / eig)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0 and '"l_max": 1' in r.stdout
