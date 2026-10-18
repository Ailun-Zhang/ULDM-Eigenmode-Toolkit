"""Review fixes R9 (2026-10-18): inline recorder, --from-inline and pipeline robustness."""
import hashlib
import json
import os
import shutil

import h5py
import numpy as np
import pytest

from conftest import write_run
from _legacy_chain import assert_h5_bitwise
from Eig_ULDM_packages import pipeline
from Eig_ULDM_packages.cnlm_inline import CnlmInlineSettings, InlineCnlmRecorder
from Eig_ULDM_packages.cnlm_snapshot import KERNEL_FORMAT_VERSION, kernel_fingerprint
from Eig_ULDM_packages.pyul_run import PyULRun

KW = dict(l_max=2, n_shells=13, num_points=24)


def _record(run, eigen_file, n_files=None, **kw):
    st = CnlmInlineSettings.validate(str(run), eigen_file=eigen_file, **kw)
    rec = InlineCnlmRecorder(st, default_threads=2).setup()
    files = sorted((run / "Outputs" / "3WfnRS").glob("P3R_#*.npy"))[:n_files]
    t = np.linspace(0.0, 30.0, len(files))
    for k, f in enumerate(files):
        rec.record(k, np.load(f), t[k])
    return st, rec, t


def _pipe(*argv):
    return pipeline.run_pipeline(pipeline.build_parser().parse_args([str(a) for a in argv]))


# --------------------------------------------------------------------------- R9-05 (CNLM-1)
def test_eigen_file_inside_outputs_is_refused_before_anything_is_deleted(tmp_path, small_eigen_file):
    run = write_run(tmp_path / "run", n=16, n_snap=2)
    for sub in ("Cnlm", "cnlm_pipeline"):        # previous auto file / default --build-eigen location
        inside = run / "Outputs" / sub / "eigs.h5"
        inside.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(small_eigen_file, inside)
        with pytest.raises(ValueError, match="lies inside .*Outputs.*Nothing was deleted"):
            CnlmInlineSettings.validate(str(run), eigen_file=str(inside), l_max=2, num_points=16, overwrite=True)
        assert inside.is_file()
        st = CnlmInlineSettings.validate(str(run), eigen_file=str(inside), l_max=2, num_points=16,
                                         outputs_replaced=False)
        assert st.eigen_file == str(inside)
    outside = run / "eigs_copy.h5"                 # the documented remedy: a copy outside Outputs
    shutil.copy(small_eigen_file, outside)
    assert CnlmInlineSettings.validate(str(run), eigen_file=str(outside), l_max=2, num_points=16).eigen_file


# --------------------------------------------------------------------------- R9-06 (CNLM-5)
def test_recorder_keeps_one_complex_vector_per_save(tmp_path, small_eigen_file):
    run = write_run(tmp_path / "run", n=24, n_snap=3)
    st, rec, t = _record(run, small_eigen_file, **KW)
    n_modes = len(rec.projector.keys)
    assert all(isinstance(v, np.ndarray) and v.dtype == np.complex128 and v.shape == (n_modes,)
               for v in rec._cnlm)
    out = rec.finalize(t)["cnlm_file"]
    res = _pipe(run, "--eigen-file", small_eigen_file, "--num-points", 24, "--n-shells", 13, "--l-max", 2,
                "--workers", 2, "--out-dir", tmp_path / "pipe")
    assert_h5_bitwise(res["cnlm_file"], out)


# --------------------------------------------------------------------------- R9-07 (CNLM-3)
def test_fingerprint_does_not_depend_on_the_package_version():
    kw = dict(input_prefix="3WfnRS/P3R_#", dx=0.2, rs_side=5.0, rs_resol=25, length_units="kpc",
              r_values=np.linspace(1e-5, 2.5, 9), l_max=4, num_points=16, angular_grid="legacy")
    fp_a, h_a = kernel_fingerprint(toolkit_version="0.3.0", **kw)
    fp_b, h_b = kernel_fingerprint(toolkit_version="9.9.9", **kw)
    assert fp_a == fp_b and h_a == h_b and fp_a["toolkit"] == KERNEL_FORMAT_VERSION == "0.3.0"
    # identical to the 0.3.0 hash (which hashed the package version 0.3.0), so 0.3.0 checkpoints stay valid
    r = np.ascontiguousarray(kw["r_values"])
    old = {"toolkit": "0.3.0", "input_prefix": kw["input_prefix"], "dx": 0.2, "rs_side": 5.0, "rs_resol": 25,
           "length_units": "kpc", "r_values_sha256": hashlib.sha256(r.tobytes()).hexdigest(), "n_r": 9,
           "l_max": 4, "num_points": 16, "angular_grid": "legacy", "center": "com"}
    assert h_a == hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()[:16]


def test_from_inline_after_version_bump_and_failed_force_keeps_params(tmp_path, small_eigen_file, monkeypatch):
    run = write_run(tmp_path / "run", n=24, n_snap=3)
    st, rec, t = _record(run, small_eigen_file, **KW)
    ref = rec.finalize(t)["cnlm_file"]
    shutil.copy(ref, tmp_path / "ref.h5")
    os.remove(ref)
    shutil.rmtree(run / "Outputs" / "3WfnRS")      # inline checkpoints only
    monkeypatch.setattr(pipeline, "__version__", "0.3.1")
    res = _pipe(run, "--from-inline")
    assert_h5_bitwise(res["cnlm_file"], str(tmp_path / "ref.h5"))
    params = run / "Outputs" / "Cnlm" / "pipeline_params.json"
    before = params.read_bytes()
    assert json.loads(before)["toolkit_version"] == "0.3.1"
    # a --force call with other kernel parameters cannot recompute inline checkpoints: it fails, and the
    # parameter record of the good checkpoints is left as it was
    with pytest.raises(SystemExit, match="cannot be recomputed"):
        _pipe(run, "--from-inline", "--force", "--num-points", 16)
    assert params.read_bytes() == before
    assert os.path.exists(res["cnlm_file"])
    os.remove(res["cnlm_file"])
    _pipe(run, "--from-inline")                    # still usable without --force


# --------------------------------------------------------------------------- R9-08 (NUM-4)
def test_from_inline_refuses_a_replaced_eigen_file(tmp_path, small_eigen_file):
    run = write_run(tmp_path / "run", n=24, n_snap=2)
    eig = tmp_path / "shared_eigs.h5"
    shutil.copy(small_eigen_file, eig)
    st, rec, t = _record(run, str(eig), **KW)
    ref = rec.finalize(t)["cnlm_file"]
    os.remove(ref)
    with h5py.File(eig, "a") as f:                 # "rebuilt" after the run: same r_max, other bytes
        f.attrs["rebuilt"] = 1
    with pytest.raises(SystemExit, match="not the one the inline run used"):
        _pipe(run, "--from-inline")
    assert not os.path.exists(ref)
    with pytest.warns(RuntimeWarning, match="not the one the inline run used"):
        _pipe(run, "--from-inline", "--force")
    os.remove(ref)
    _pipe(run, "--from-inline", "--eigen-file", eig)   # an explicit choice is accepted


# --------------------------------------------------------------------------- R9-09 (CNLM-2)
def _save_every_step_run(tmp_path, n_snap=3, times_file=True):
    run = write_run(tmp_path / "run", n=24, n_snap=n_snap, save_number=-1)
    if times_file:
        np.save(run / "Outputs" / "snapshot_times_Myr.npy", np.array([0.0, 5.0, 10.0])[:n_snap])
    return run


def test_effective_save_number_for_save_every_step(tmp_path):
    run = PyULRun.open(_save_every_step_run(tmp_path))
    assert run.save_number == -1 and run.effective_save_number() == 2
    assert run.effective_save_number(extra_indices=[5]) == 5
    assert list(run.snapshot_times_myr([0, 2])) == [0.0, 10.0]
    with pytest.raises(ValueError, match="effective save number"):
        pipeline.parse_snapshot_spec(None, -1)


def test_pipeline_and_from_inline_with_save_every_step(tmp_path, small_eigen_file):
    run = _save_every_step_run(tmp_path)
    st, rec, t = _record(run, small_eigen_file, **KW)
    ref = rec.finalize(np.array([0.0, 5.0, 10.0]))["cnlm_file"]
    shutil.copy(ref, tmp_path / "ref.h5")
    os.remove(ref)
    res = _pipe(run, "--from-inline")
    assert_h5_bitwise(res["cnlm_file"], str(tmp_path / "ref.h5"))
    res2 = _pipe(run, "--eigen-file", small_eigen_file, "--num-points", 24, "--n-shells", 13, "--l-max", 2,
                 "--out-dir", tmp_path / "pipe")
    with h5py.File(res2["cnlm_file"], "r") as f:
        assert list(f["t"][()]) == [0.0, 5.0, 10.0]
        assert list(f.attrs["snapshot_indices"]) == [0, 1, 2]


def test_save_every_step_without_times_file_fails_clearly(tmp_path, small_eigen_file):
    run = _save_every_step_run(tmp_path, times_file=False)
    with pytest.raises(SystemExit, match="snapshot_times_Myr.npy"):
        _pipe(run, "--eigen-file", small_eigen_file, "--num-points", 24, "--n-shells", 13, "--l-max", 2,
              "--out-dir", tmp_path / "pipe")
