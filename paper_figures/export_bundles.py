#!/usr/bin/env python3
"""Write one single-file trajectory bundle (runs/<run>.npz) per run of a config that lists raw run directories.

A bundle replaces the ~1500 per-save files of a run directory (NBody/ULDCOM/DF) with one compressed .npz that
also carries the energies, ULDM mass, particle masses and the verbatim config.uldm / reproducibility.uldm.
The figure scripts read bundles transparently (``--prefer bundle``, or automatically when the raw run directory
is absent).  Runs whose raw directory is missing, or that have no raw ``path`` in the config, are skipped and
reported.  The bundle path is the run's ``bundle`` entry, written under ``--out-root`` (default: the data root).

The default config describes the deposit, which has bundles only; to build bundles from raw PyUL run directories
use the raw-layout config (or your own copy of it):

    python paper_figures/export_bundles.py --config paper_figures/figures_config.raw_example.json
        --data-root RAW_ROOT --out-root DEPOSIT_ROOT [--runs fig1_m1 ...] [--overwrite] [--verify]

(one command line; wrapped here for width)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from Eig_ULDM_packages import trajectories as traj  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default=str(HERE / "figures_config.json"))
    parser.add_argument("--out-root", default=None,
                        help="root under which each run's 'bundle' path is written (default: data root)")
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--axion-mass-ev", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true", help="reload each bundle and compare with the raw run")
    args = parser.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root) if args.out_root else data_root
    report = {}
    for key in args.runs or sorted(cfg["runs"]):
        entry = cfg["runs"][key]
        if not entry.get("path"):
            report[key] = "skipped: no raw 'path' in this config"
            print(f"[{key}] {report[key]}")
            continue
        raw = data_root / entry["path"]
        out = out_root / entry["bundle"]
        if not raw.exists():
            report[key] = "skipped: raw run directory missing"
            print(f"[{key}] {report[key]}")
            continue
        run = traj.load_run(raw, axion_mass_ev=args.axion_mass_ev)
        traj.export_bundle(run, out, overwrite=args.overwrite,
                           extra_metadata={"config_key": key, "original_run_id": entry.get("original_run_id")})
        msg = f"{out} ({out.stat().st_size / 1e6:.2f} MB, {run.n_frames} frames)"
        if args.verify:
            back = traj.load_run(out)
            same = all(
                (a is None and b is None) or (a is not None and b is not None and a.shape == b.shape and (a == b).all())
                for a, b in ((run.nbody, back.nbody), (run.uldcom, back.uldcom), (run.df, back.df),
                             (run.uldm_mass, back.uldm_mass), (run.times_myr, back.times_myr)))
            same = same and run.config_text == back.config_text and run.reproducibility_text == back.reproducibility_text
            same = same and sorted(run.energies) == sorted(back.energies) and all(
                (run.energies[k] == back.energies[k]).all() for k in run.energies)
            msg += " verified" if same else " VERIFY FAILED"
            if not same:
                report[key] = "verify failed"
        report.setdefault(key, "ok")
        print(f"[{key}] {msg}")
    bad = [k for k, v in report.items() if v not in ("ok",) and not v.startswith("skipped")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
