"""Tests for Eig_ULDM_packages.trajectories (synthetic PyUL run directories; real data optional)."""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Eig_ULDM_packages import trajectories as T  # noqa: E402

PC = 12133.095824299327  # pc per code length at 1e-21 eV


def make_run(root, *, n_saves=10, duration=3000, start=0.0, n_bodies=2, fmt="npy", layout="new",
             repro=None, snapshot=None, name="20250101_000000@64"):
    run = Path(root) / name
    out = run / "Outputs"
    out.mkdir(parents=True)
    config = {
        "PyUL Version": 30.5,
        "Save Options": {"Flags": "Energy DF ULDCOM NBody", "Number": n_saves, "Format": fmt},
        "Spatial Resolution": 64,
        "Duration": {"Time Duration": duration, "Start Time": start, "Time Units": "Myr"},
        "Simulation Box": {"Box Length": 5.0, "Length Units": "kpc"},
        "ULDM Solitons": {"Condition": [], "Mass Units": "M_solar_masses"},
        "Matter Particles": {"Condition": [[0.24, [0.14, 0, 0], [0, 6.5, 0]]] * n_bodies,
                             "Mass Units": "M_solar_masses", "Position Units": "kpc", "Velocity Units": "km/s"},
    }
    if repro is None:
        repro = {"Integrator Version": 30.5, "Axion Mass": 1.7826619099999998e-57, "m22": 10.0}
    (run / "config.uldm").write_text(json.dumps(config, indent=4), encoding="utf-8")
    (run / "reproducibility.uldm").write_text(json.dumps(repro, indent=4), encoding="utf-8")
    stems = {"NBody": "NTM", "ULDCOM": "UCM", "DF": "DYF"}
    data = {"NBody": [], "ULDCOM": [], "DF": []}
    for k in range(n_saves + 1):
        nb = np.zeros(6 * n_bodies)
        for b in range(n_bodies):
            nb[6 * b:6 * b + 3] = [0.01 * (b + 1) + 1e-4 * k, 2e-4 * k, 0.0]
            nb[6 * b + 3:6 * b + 6] = [0.0, 1.0 + k, 0.0]
        com = np.array([1e-4 * k, 0.0, 0.0])
        df = np.full(3 * n_bodies, float(k))
        for typ, arr in (("NBody", nb), ("ULDCOM", com), ("DF", df)):
            data[typ].append(arr)
            folder = out / typ if layout == "new" else out
            folder.mkdir(exist_ok=True)
            path = folder / f"{stems[typ]}_#{k:03d}.{fmt}"
            if fmt == "npy":
                np.save(path, arr)
            elif fmt == "npz":
                np.savez(path, arr)
            else:
                h5py = pytest.importorskip("h5py")
                with h5py.File(path, "w") as f:
                    f.create_dataset(typ, data=arr)
    np.save(out / "egylist.npy", np.arange(n_saves + 1, dtype=float))
    np.save(out / "masseslist.npy", np.ones(n_saves + 1))
    np.save(out / "ULDMass.npy", np.full(n_saves + 1, 170.0))
    if snapshot is not None:
        np.save(out / snapshot[0], np.asarray(snapshot[1]))
    return run, {k: np.vstack(v) for k, v in data.items()}


def test_length_conversion_constant():
    assert T.verify_length_conversion() == pytest.approx(12133.0958, abs=5e-5)
    assert T.pc_per_code_length(1e-21) == pytest.approx(PC, rel=1e-14)
    # length unit scales as m^{-1/2}
    assert T.pc_per_code_length(1e-22) == pytest.approx(PC * np.sqrt(10.0), rel=1e-12)
    # 12e6 Msun is the M_code = 170.3149822613627 used by the c_nlm normalisation
    assert T.convert_to_code(12, "M_solar_masses", "m", 1e-21) == pytest.approx(170.3149822613627, rel=1e-13)
    assert T.convert_from_code(T.convert_to_code(3000.0, "Myr", "t"), "Myr", "t") == pytest.approx(3000.0)


def test_time_axis_correct_and_legacy(tmp_path):
    run_dir, _ = make_run(tmp_path, n_saves=500, duration=3000)
    run = T.load_run(run_dir)
    assert run.n_frames == 501
    assert run.times_myr[0] == 0.0 and run.times_myr[-1] == pytest.approx(3000.0)
    assert np.allclose(np.diff(run.times_myr), 6.0)
    legacy = T.load_run(run_dir, legacy_time_axis=True)
    assert legacy.times_myr[-1] == pytest.approx(500 * 3000 / 501)  # 2994.0119...
    assert run.with_time_axis(True).times_myr[-1] == pytest.approx(legacy.times_myr[-1])


def test_start_time_offset(tmp_path):
    run_dir, _ = make_run(tmp_path, n_saves=4, duration=100, start=50.0)
    run = T.load_run(run_dir)
    assert np.allclose(run.times_myr, [50, 75, 100, 125, 150])
    # the legacy convention ignored the start time, as the old notebooks did
    assert np.allclose(T.load_run(run_dir, legacy_time_axis=True).times_myr, np.arange(5) * 100 / 5)


def test_observables(tmp_path):
    run_dir, raw = make_run(tmp_path, n_saves=6)
    run = T.load_run(run_dir)
    assert run.n_bodies == 2
    for b in (0, 1):
        pos = raw["NBody"][:, 6 * b:6 * b + 3]
        assert np.allclose(run.distance_from_origin_pc(b), np.linalg.norm(pos, axis=1) * PC)
        assert np.allclose(run.distance_from_uldm_com_pc(b), np.linalg.norm(pos - raw["ULDCOM"], axis=1) * PC)
        assert np.allclose(run.positions_pc(b), pos * PC)
    assert np.allclose(run.uldcom_pc(), raw["ULDCOM"] * PC)
    assert run.particle_masses_msun() == pytest.approx([240000.0, 240000.0])
    assert run.energies["egylist"].shape == (7,)
    assert run.uldm_mass.shape == (7,)
    with pytest.raises(IndexError):
        run.distance_from_origin_pc(2)


def test_axion_mass_from_run_not_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYUL_AXION_MASS_EV", "1e-22")
    run_dir, raw = make_run(tmp_path)
    with pytest.warns(RuntimeWarning, match="recorded value"):
        run = T.load_run(run_dir)
    assert run.axion_mass_ev == 1e-21 and run.axion_mass_source == "reproducibility.uldm"
    assert run.pc_per_code == pytest.approx(PC)
    with pytest.raises(ValueError):
        T.load_run(run_dir, axion_mass_ev=1e-22)  # disagrees with the run record
    assert T.load_run(run_dir, axion_mass_ev=1e-21).axion_mass_ev == 1e-21


def test_axion_mass_missing_requires_argument(tmp_path, monkeypatch):
    # F15 policy: record > explicit argument > PYUL_AXION_MASS_EV (last resort, warned) > error
    monkeypatch.delenv("PYUL_AXION_MASS_EV", raising=False)
    run_dir, _ = make_run(tmp_path, repro={"Integrator Version": 30.5})
    with pytest.raises(ValueError, match="axion"):
        T.load_run(run_dir)
    monkeypatch.setenv("PYUL_AXION_MASS_EV", "1e-22")
    run = T.load_run(run_dir, axion_mass_ev=1e-21)          # the argument beats the environment
    assert run.axion_mass_ev == 1e-21 and run.axion_mass_source == "argument"
    with pytest.warns(RuntimeWarning, match="last resort"):
        run = T.load_run(run_dir)
    assert run.axion_mass_ev == 1e-22 and "PYUL_AXION_MASS_EV" in run.axion_mass_source


def test_inconsistent_axion_record(tmp_path):
    run_dir, _ = make_run(tmp_path, repro={"Axion Mass": 1.7826619099999998e-57, "m22": 1.0})
    with pytest.raises(ValueError, match="Inconsistent"):
        T.load_run(run_dir)


@pytest.mark.parametrize("fmt,layout", [("npy", "old"), ("npz", "new"), ("hdf5", "new")])
def test_formats_and_layouts(tmp_path, fmt, layout):
    run_dir, raw = make_run(tmp_path, fmt=fmt, layout=layout, n_saves=3)
    run = T.load_run(run_dir)
    assert np.array_equal(run.nbody, raw["NBody"])
    assert np.array_equal(run.uldcom, raw["ULDCOM"])
    assert np.array_equal(run.df, raw["DF"])


def test_bundle_roundtrip(tmp_path):
    run_dir, _ = make_run(tmp_path / "runs", n_saves=8)
    run = T.load_run(run_dir)
    out = T.export_bundle(run, tmp_path / "b" / "x.npz")
    with pytest.raises(FileExistsError):
        T.export_bundle(run, out)
    back = T.load_run(out)
    assert back.is_bundle
    for name in ("nbody", "uldcom", "df", "uldm_mass", "times_myr", "frame_index"):
        assert np.array_equal(getattr(run, name), getattr(back, name)), name
    assert back.config_text == run.config_text and back.reproducibility_text == run.reproducibility_text
    assert sorted(back.energies) == sorted(run.energies)
    assert back.axion_mass_ev == run.axion_mass_ev
    assert np.allclose(T.load_run(out, legacy_time_axis=True).times_myr, np.arange(9) * 3000 / 9)
    with np.load(out, allow_pickle=False) as z:  # no pickled objects inside
        assert str(z["format"]) == T.BUNDLE_FORMAT


def test_bundle_keeps_explicit_axion_mass(tmp_path):
    run_dir, _ = make_run(tmp_path / "runs", repro={"Integrator Version": 26.32})
    out = T.export_bundle(run_dir, tmp_path / "x.npz", axion_mass_ev=1e-21)
    back = T.load_bundle(out)
    assert back.axion_mass_ev == 1e-21 and "bundle metadata" in back.axion_mass_source


def test_snapshot_times_file(tmp_path):
    run_dir, _ = make_run(tmp_path, n_saves=4, duration=100, snapshot=("snapshot_times_Myr.npy", [0, 25, 50, 75, 100]))
    run = T.load_run(run_dir)
    assert run.time_axis_source.endswith("snapshot_times_Myr.npy")
    assert np.allclose(run.times_myr, [0, 25, 50, 75, 100])
    run_dir2, _ = make_run(tmp_path / "b", n_saves=4, duration=100,
                           snapshot=("snapshot_times_Myr.npy", [0, 20, 40, 60, 80]))
    with pytest.warns(RuntimeWarning, match="deviates"):
        run2 = T.load_run(run_dir2)
    assert np.allclose(run2.times_myr, [0, 20, 40, 60, 80])
    # the legacy option always uses the old-notebook formula
    assert np.allclose(T.load_run(run_dir2, legacy_time_axis=True).times_myr, np.arange(5) * 100 / 5)
    # code-unit snapshot file
    t_code = T.convert_to_code(np.array([0.0, 25, 50, 75, 100]), "Myr", "t")
    run_dir3, _ = make_run(tmp_path / "c", n_saves=4, duration=100, snapshot=("snapshot_times_code.npy", t_code))
    assert np.allclose(T.load_run(run_dir3).times_myr, [0, 25, 50, 75, 100])


def test_find_run_dir(tmp_path):
    run_dir, _ = make_run(tmp_path / "save_path")
    assert T.find_run_dir(tmp_path / "save_path") == run_dir
    make_run(tmp_path / "save_path", name="20250102_000000@64")
    with pytest.raises(ValueError, match="several run folders"):
        T.find_run_dir(tmp_path / "save_path")


def test_noncontiguous_frames_raise(tmp_path):
    run_dir, _ = make_run(tmp_path, n_saves=5)
    os.remove(run_dir / "Outputs" / "NBody" / "NTM_#002.npy")
    with pytest.raises(ValueError, match="contiguous"):
        T.load_run(run_dir)


def test_window(tmp_path):
    run_dir, raw = make_run(tmp_path, n_saves=10, duration=100)
    run = T.load_run(run_dir).window(0, 50)
    assert run.n_frames == 6 and run.times_myr[-1] == pytest.approx(50)
    assert np.array_equal(run.nbody, raw["NBody"][:6])
    assert run.energies["egylist"].shape == (6,)


# ULDM_PAPER_DATA_ROOT = unpacked data deposit (StoneSkipping_PRD_data_v1); the binary run is runs/binary256.npz
DATA_ROOT = Path(os.environ.get("ULDM_PAPER_DATA_ROOT", "")).expanduser()
BINARY = DATA_ROOT / "runs" / "binary256.npz"


@pytest.mark.skipif(not (os.environ.get("ULDM_PAPER_DATA_ROOT") and BINARY.is_file()),
                    reason="paper data not available")
def test_real_binary_run():
    run = T.load_run(BINARY)
    assert run.n_frames == 501 and run.n_bodies == 2
    assert run.axion_mass_ev == 1e-21
    assert run.distance_from_origin_pc(0)[0] == pytest.approx(140.0, rel=1e-12)
    assert run.times_myr[-1] == 3000.0
    assert T.load_run(BINARY, legacy_time_axis=True).times_myr[-1] == pytest.approx(2994.011976047904)
