"""Tests for the paper_figures scripts.  Data-dependent checks are skipped unless ULDM_PAPER_DATA_ROOT points
to the unpacked data deposit (StoneSkipping_PRD_data_v1, the layout of paper_figures/figures_config.json)."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "paper_figures"
for p in (str(REPO), str(FIGDIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

CONFIG = json.loads((FIGDIR / "figures_config.json").read_text(encoding="utf-8"))
DATA_ROOT = Path(os.environ.get("ULDM_PAPER_DATA_ROOT", "")).expanduser()
HAVE_DATA = bool(os.environ.get("ULDM_PAPER_DATA_ROOT")) and (DATA_ROOT / "runs").is_dir()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="paper data not available")


def _ctx(fig_key, extra=()):
    import _common

    args = _common.make_parser(fig_key, "test").parse_args(["--data-root", str(DATA_ROOT), "--no-provenance", *extra])
    return _common.Context.from_args(args)


def test_config_paths_are_relative_and_complete():
    for section in ("runs", "files"):
        for key, entry in CONFIG[section].items():
            for field in ("path", "bundle"):
                rel = entry.get(field)
                if rel is None:
                    continue
                assert not os.path.isabs(rel) and ":" not in rel and not rel.startswith(("/", "\\")), (key, rel)
                assert ".." not in Path(rel).parts, (key, rel)
    keys = sorted(CONFIG["figures"])
    assert keys == ["fig%02d" % i for i in range(1, 13)]
    for key, fig in CONFIG["figures"].items():
        assert (FIGDIR / fig["script"]).is_file(), fig["script"]
        for run in fig.get("runs", []):
            assert run in CONFIG["runs"], (key, run)
        for f in fig.get("files", []):
            assert f in CONFIG["files"], (key, f)


@pytest.mark.parametrize("key", sorted(CONFIG["figures"]))
def test_scripts_have_common_cli(key):
    script = FIGDIR / CONFIG["figures"][key]["script"]
    out = subprocess.run([sys.executable, "-B", str(script), "--help"], capture_output=True, text=True, cwd=str(FIGDIR))
    assert out.returncode == 0, out.stderr
    for opt in ("--data-root", "--config", "--out-dir", "--formats"):
        assert opt in out.stdout, (key, opt)


def test_fig12_analytic(tmp_path):
    import fig12_resonance_roots as F12

    fp, fm = F12.roots(np.array([0.0, F12.X_LOW, F12.X_HIGH, 0.5, 2.0]))
    assert fp[0] == pytest.approx(1.0) and fm[0] == pytest.approx(1.0 / 3.0)
    assert fp[1] == pytest.approx(fm[1]) and fp[2] == pytest.approx(fm[2])  # roots coalesce at the thresholds
    assert np.isnan(fp[3]) and np.isnan(fm[3])  # complex between the thresholds
    assert fp[4] < 0 and fm[4] < 0  # large-damping branch: non-positive roots
    assert F12.X_LOW == pytest.approx(0.258819, abs=1e-6) and F12.X_HIGH == pytest.approx(0.965926, abs=1e-6)
    F12.main(["--out-dir", str(tmp_path), "--formats", "pdf,png"])
    assert (tmp_path / "f_x.pdf").is_file() and (tmp_path / "f_x.png").is_file()


@needs_data
def test_fig01_matches_published_input():
    import fig01_mass_sequence_radius as F1

    ref = json.loads((DATA_ROOT / CONFIG["files"]["derived_fig1_json"]["path"]).read_text(encoding="utf-8"))["series"]
    series = F1.compute(_ctx("fig01"))
    for mine, theirs in zip(series, ref):
        assert mine["mass_percent"] == theirs["mass_percent"]
        assert np.array_equal(mine["time_Myr"], np.asarray(theirs["time_Myr"], dtype=float))
        assert np.max(np.abs(mine["radius_pc"] - np.asarray(theirs["radius_pc"]))) < 1e-9


@needs_data
def test_fig02_raw_matches_published_csv_for_192_and_256():
    import fig02_resolution_robustness as F2

    ctx = _ctx("fig02")
    raw = F2.compute_raw(ctx, 3000.0, {"128": "res128_jan2026", "192": "res192", "256": "res256"})
    csvd = F2.compute_csv(ctx, 3000.0)
    for res in ("192", "256"):
        assert np.max(np.abs(raw[res]["r_sys_pc"] - csvd[res]["r_sys_pc"])) < 1e-9
        assert np.max(np.abs(raw[res]["Pdip_nle2"] - csvd[res]["Pdip_nle2"])) < 1e-15
    # the published N=128 curve is from the 128_new rerun, not from the January 2026 run in the extract
    assert np.max(np.abs(raw["128"]["r_sys_pc"] - csvd["128"]["r_sys_pc"])) > 0.01


@needs_data
def test_fig11_peaks_match_manuscript():
    import fig11_cnlm_time_spectrum as F11
    import _common

    p = _common.make_parser("fig11", "x")
    p.add_argument("--mode", type=int, nargs=3, default=[1, 1, -1])
    p.add_argument("--time-sampling", default="dense")
    p.add_argument("--n-dense", type=int, default=60001)
    p.add_argument("--analysis-t-max", type=float, default=None)
    p.add_argument("--analysis-mode", default="complex")
    p.add_argument("--peak-rel-threshold", type=float, default=0.01)
    args = p.parse_args(["--data-root", str(DATA_ROOT), "--no-provenance"])
    result, _ = F11.compute(_ctx("fig11"), args)
    f2, f1 = sorted(f for f, _ in result.strong_peaks_2)
    assert f1 == pytest.approx(0.000999983, abs=5e-10)
    assert f2 == pytest.approx(-0.000233329, abs=5e-10)


@needs_data
def test_legacy_axis_figures_end_at_2994(tmp_path):
    import fig07_reconstructed_background as F7

    series = F7.compute(_ctx("fig07"), "compare_n0", True)
    for _, d in series:
        assert d["time_Myr"][-1] == pytest.approx(500 * 3000 / 501)


def test_make_all_figures_relative_out_dir_from_other_cwd(tmp_path):
    """R9-01: relative --out-dir/--config are relative to the caller's cwd, not to paper_figures/."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "cfg").mkdir()
    (work / "cfg" / "figures_config.json").write_text((FIGDIR / "figures_config.json").read_text(encoding="utf-8"),
                                                      encoding="utf-8")
    out = subprocess.run([sys.executable, "-B", str(FIGDIR / "make_all_figures.py"), "--config",
                          os.path.join("cfg", "figures_config.json"), "--out-dir", "figs", "--only", "fig12",
                          "--formats", "png"], capture_output=True, text=True, cwd=str(work))
    assert out.returncode == 0, out.stdout + out.stderr
    assert (work / "figs" / "f_x.png").is_file()
    assert (work / "figs" / "make_all_figures_status.json").is_file()
    assert not (FIGDIR / "figs").exists()


@needs_data
def test_make_all_figures_relative_data_root_from_other_cwd(tmp_path):
    """R9-01: the deposit README command (relative --data-root) works from the data root's parent folder."""
    root = DATA_ROOT.resolve()
    out = subprocess.run([sys.executable, "-B", str(FIGDIR / "make_all_figures.py"), "--data-root", root.name,
                          "--out-dir", str(tmp_path / "figs"), "--only", "fig01", "--formats", "png"],
                         capture_output=True, text=True, cwd=str(root.parent))
    assert out.returncode == 0, out.stdout + out.stderr
    assert (tmp_path / "figs" / "distance_from_origin_vs_time_v7_256.png").is_file()
