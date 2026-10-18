#!/usr/bin/env python3
"""Regenerate every printed figure of the paper (Figs. 1-12) with default (= published) options.

Each figure runs in its own Python process so that the matplotlib rcParams of one script cannot leak into
another.  A figure that fails (e.g. missing data) is reported and the others still run; the exit status is
non-zero if any failed.

    python paper_figures/make_all_figures.py --data-root /path/to/archive --out-dir figures_out
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config", default=str(HERE / "figures_config.json"))
    parser.add_argument("--out-dir", default="paper_figures_output")
    parser.add_argument("--formats", default=None)
    parser.add_argument("--prefer", choices=("raw", "bundle"), default="raw")
    parser.add_argument("--only", nargs="+", default=None, help="figure keys, e.g. fig01 fig07")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="extra arguments passed to every script (put last)")
    args = parser.parse_args(argv)
    # Relative paths mean "relative to the caller's working directory", exactly as for a single figure script.
    # Resolve them here so that the child processes (and the status file) all see the same absolute paths.
    args.config = str(Path(args.config).expanduser().resolve())
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    if args.data_root:
        args.data_root = str(Path(args.data_root).expanduser().resolve())
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    keys = args.only or sorted(cfg["figures"])
    status = {}
    for key in keys:
        script = HERE / cfg["figures"][key]["script"]
        cmd = [sys.executable, "-B", str(script), "--config", args.config, "--out-dir", args.out_dir,
               "--prefer", args.prefer]
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        if args.formats:
            cmd += ["--formats", args.formats]
        cmd += list(args.extra)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        status[key] = {"returncode": proc.returncode, "seconds": round(time.time() - t0, 1)}
        print(f"[{key}] {'ok' if proc.returncode == 0 else 'FAILED'} ({status[key]['seconds']} s)")
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
            status[key]["error"] = tail
            print("    " + "\n    ".join(tail))
    failed = [k for k, v in status.items() if v["returncode"]]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "make_all_figures_status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)
    print(f"{len(keys) - len(failed)}/{len(keys)} figures written to {args.out_dir}" +
          (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
