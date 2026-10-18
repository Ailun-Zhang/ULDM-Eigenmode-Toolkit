"""WP3: in-simulation recorder (cnlm_inline), pipeline --from-inline, and the single run loader (F15)."""
import json
import os
import shutil
import warnings

import h5py
import numpy as np
import pytest

from conftest import write_run
from _legacy_chain import assert_h5_bitwise
from Eig_ULDM_packages import pipeline, pyul_run
from Eig_ULDM_packages.cnlm_inline import CnlmInlineSettings, InlineCnlmRecorder
from Eig_ULDM_packages.c_nlm_integrator import load_cnlm_from_h5


def _record_run(run, eigen_file, **kw):
    """Drive the recorder as PyUL does, with the cubes of the synthetic run's 3WfnRS files."""
    st = CnlmInlineSettings.validate(str(run), eigen_file=eigen_file, **kw)
    rec = InlineCnlmRecorder(st, default_threads=2).setup()
    files = sorted((run / "Outputs" / "3WfnRS").glob("P3R_#*.npy"))
    t = np.linspace(0.0, 30.0, len(files))
    for k, f in enumerate(files):
        rec.record(k, np.load(f), t[k])
    return st, rec, t


def test_recorder_equals_pipeline_bitwise(tmp_path, small_eigen_file):
    run = write_run(tmp_path / "run", n=24, n_snap=4)
    kw = dict(l_max=2, n_shells=13, num_points=24, save_alm=True)
    st, rec, t = _record_run(run, small_eigen_file, **kw)
    res_inline = rec.finalize(t)
    args = [str(run), "--eigen-file", small_eigen_file, "--num-points", "24", "--n-shells", "13", "--l-max", "2",
            "--workers", "3", "--out-dir", str(tmp_path / "pipe"), "--alm-out", "alm.h5"]
    res = pipeline.run_pipeline(pipeline.build_parser().parse_args(args))
    assert_h5_bitwise(res["cnlm_file"], res_inline["cnlm_file"])
    assert_h5_bitwise(res["alm_file"], res_inline["alm_file"])
    with h5py.File(res_inline["cnlm_file"], "r") as f:
        assert sorted(f.keys()) == ["n", "t"]
        assert f.attrs["source"] == "pyul-inline" and f.attrs["fingerprint"] == rec.fp_hash
    rec_json = json.loads((run / "Outputs" / "Cnlm" / "cnlm_inline.json").read_text())
    assert rec_json["status"] == "finished" and rec_json["snapshots_recorded"] == 4


def test_from_inline_assembles_after_a_crash(tmp_path, small_eigen_file):
    run = write_run(tmp_path / "run", n=24, n_snap=4)
    st, rec, t = _record_run(run, small_eigen_file, l_max=2, n_shells=13, num_points=24)
    ref = rec.finalize(t)["cnlm_file"]
    ref_copy = str(tmp_path / "ref.h5")
    shutil.copy(ref, ref_copy)
    os.remove(ref)
    # the "crash": no final file, and no 3WfnRS files at all (inline runs need not save them)
    shutil.rmtree(run / "Outputs" / "3WfnRS")
    res = pipeline.run_pipeline(pipeline.build_parser().parse_args([str(run), "--from-inline"]))
    assert res["cnlm_file"] == ref
    t1, c1, a1 = load_cnlm_from_h5(ref)
    t2, c2, _ = load_cnlm_from_h5(ref_copy)
    # same series bit for bit; the time axis of the crashed run comes from the run (no snapshot_times file
    # in this synthetic run -> linspace(0, duration, save_number+1)), identical here
    assert sorted(c1) == sorted(c2) and all(np.array_equal(c1[k], c2[k]) for k in c1)
    assert np.array_equal(t1, t2) and a1["producer"] == "Eig_ULDM_packages.pipeline"
    # a partial run: only the first two checkpoints survived
    os.remove(ref)
    for k in (2, 3):
        os.remove(run / "Outputs" / "Cnlm" / "alm_checkpoints" / f"ALM_#{k:03d}.npz")
    res = pipeline.run_pipeline(pipeline.build_parser().parse_args([str(run), "--from-inline"]))
    t3, c3, _ = load_cnlm_from_h5(res["cnlm_file"])
    assert t3.size == 2 and all(np.array_equal(c3[k], c2[k][:2]) for k in c2)


def test_validation_errors(tmp_path, small_eigen_file):
    run = write_run(tmp_path / "run", n=16, n_snap=2)
    with pytest.raises(ValueError, match="lmax=2"):
        CnlmInlineSettings.validate(str(run), eigen_file=small_eigen_file, l_max=3, num_points=16)
    with pytest.raises(ValueError, match="even"):
        CnlmInlineSettings.validate(str(run), eigen_file=small_eigen_file, l_max=2, num_points=15)
    with pytest.raises(ValueError, match="UseInit"):
        CnlmInlineSettings.validate(str(run), use_init=True)
    st = CnlmInlineSettings.validate(str(run), use_init=True, soliton_mass=12.0)
    assert st.eigen_file is None and st.soliton_mass == 12.0
    cfg = json.loads((run / "config.uldm").read_text())
    cfg["ULDM Solitons"]["Condition"].append([6.0, [1, 0, 0], [0, 0, 0], 0.0])
    (run / "config.uldm").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="2 standard and 0 embedded"):
        CnlmInlineSettings.validate(str(run))
    out = run / "Outputs" / "c_nlm.h5"
    out.write_bytes(b"x")
    with pytest.raises(FileExistsError, match="OverwriteOutputs"):
        CnlmInlineSettings.validate(str(run), soliton_mass=12.0)
    assert CnlmInlineSettings.validate(str(run), soliton_mass=12.0, overwrite=True).output == str(out)
    # r_max of the eigen file (2.5 kpc) inconsistent with the shells (1.5 kpc): refused in setup
    st = CnlmInlineSettings.validate(str(run), eigen_file=small_eigen_file, l_max=2, num_points=16, r_max=1.5,
                                     n_shells=7, output="c2.h5")
    with pytest.raises(ValueError, match="r_max"):
        InlineCnlmRecorder(st).setup()


def test_auto_eigen_file_matches_pipeline_build(tmp_path):
    run = write_run(tmp_path / "run", n=16, n_snap=2)
    ep = dict(n_grid=96, n_eig=2)
    st = CnlmInlineSettings.validate(str(run), l_max=1, num_points=16, n_shells=9, eigen_params=ep)
    rec = InlineCnlmRecorder(st).setup()
    assert os.path.dirname(rec.eigen_file) == st.cnlm_dir
    with h5py.File(rec.eigen_file, "r") as f:
        assert f.attrs["n_grid"] == 96 and f.attrs["lmax"] == 1 and f.attrs["rmax"] == 2.5
        assert f.attrs["axion_mass_ev"] == 1e-21 and f.attrs["soliton_mass"] == 12.0


# --------------------------------------------------------------------------- F15: one loader, one policy
def test_axion_policy_single_implementation(tmp_path, monkeypatch):
    from Eig_ULDM_packages import trajectories as T
    assert T.code_units is pyul_run.code_units and T.pyul_time_axis_myr is pyul_run.pyul_time_axis_myr
    rec = {"Axion Mass": 1.7826619099999998e-57, "m22": 10.0}
    monkeypatch.delenv("PYUL_AXION_MASS_EV", raising=False)
    assert pyul_run.resolve_axion_mass(rec) == (1e-21, "reproducibility.uldm")
    assert pyul_run.resolve_axion_mass({}, 2e-21) == (2e-21, "argument")
    with pytest.raises(ValueError, match="disagrees"):
        pyul_run.resolve_axion_mass(rec, 2e-21)
    with pytest.raises(ValueError, match="axion"):
        pyul_run.resolve_axion_mass({})
    monkeypatch.setenv("PYUL_AXION_MASS_EV", "3e-21")
    with pytest.warns(RuntimeWarning, match="last resort"):
        assert pyul_run.resolve_axion_mass({})[0] == 3e-21
    with pytest.raises(ValueError):
        pyul_run.resolve_axion_mass({}, allow_env=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pyul_run.resolve_axion_mass({}, 2e-21)[1] == "argument"   # argument beats env, no warning
    with pytest.warns(RuntimeWarning, match="recorded value"):
        assert pyul_run.resolve_axion_mass(rec)[0] == 1e-21
    # pipeline uses the same policy: the record wins over the command line value (which must agree)
    run = write_run(tmp_path / "run", n=16, n_snap=2)
    r = pyul_run.PyULRun.open(run)
    with pytest.raises(SystemExit, match="disagrees"):
        pipeline._resolve_axion_mass(r, 1e-22)
    monkeypatch.delenv("PYUL_AXION_MASS_EV")
    assert pipeline._resolve_axion_mass(r, None) == (1e-21, "reproducibility.uldm")


def test_find_run_dir_variants(tmp_path):
    run = write_run(tmp_path / "save", n=16, n_snap=1)
    for p in (run, run / "Outputs", tmp_path / "save"):
        assert pyul_run.find_run_dir(p) == str(run)
    write_run(tmp_path / "save2", n=16, n_snap=1)
    os.makedirs(tmp_path / "save2" / "other")
    shutil.copytree(tmp_path / "save2" / "20260101_000000@24", tmp_path / "save2" / "other" / "x")
    shutil.copytree(tmp_path / "save2" / "20260101_000000@24", tmp_path / "save2" / "second")
    with pytest.raises(ValueError, match="several run folders"):
        pyul_run.find_run_dir(tmp_path / "save2")
    with pytest.raises(FileNotFoundError):
        pyul_run.find_run_dir(tmp_path / "nothing")


def test_snapshot_times_prefer_pyul_file(tmp_path):
    run = write_run(tmp_path / "run", n=16, n_snap=3, save_number=2, duration=30.0)
    r = pyul_run.PyULRun.open(run)
    t, src = r.snapshot_times_myr([0, 1, 2], return_source=True)
    assert list(t) == [0.0, 15.0, 30.0] and src.startswith("linspace")
    real = np.array([0.0, 15.000000000000002, 29.999999999999996])
    np.save(run / "Outputs" / "snapshot_times_Myr.npy", real)
    t, src = r.snapshot_times_myr([0, 2], return_source=True)
    assert np.array_equal(t, real[[0, 2]]) and "snapshot_times_Myr.npy" in src
    np.save(run / "Outputs" / "snapshot_times_Myr.npy", real[:1])        # does not cover: fall back
    assert r.snapshot_times_myr([0, 2], return_source=True)[1].startswith("linspace")


def test_kernel_shell_blocks_bitwise():
    from conftest import make_cube
    from Eig_ULDM_packages.alm_kernel import AlmKernel
    cube = make_cube(20, seed=3)
    r = np.linspace(0, 2.0, 11); r[0] = 1e-5
    k = AlmKernel(r, 3, num_points=16, workers=3)
    ref, c_ref = k.compute(cube, 0.25, return_center=True)
    for block in (1, 4, 7, 11, 50):
        got, c = k.compute(cube, 0.25, return_center=True, shell_block=block)
        assert got.tobytes() == ref.tobytes() and c.tobytes() == c_ref.tobytes()
    got = k.compute(cube, 0.25, center=(1.0, 1.1, 0.9), shell_block=4)
    assert got.tobytes() == k.compute(cube, 0.25, center=(1.0, 1.1, 0.9)).tobytes()
