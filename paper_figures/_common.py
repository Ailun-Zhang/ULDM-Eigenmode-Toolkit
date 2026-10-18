"""Shared helpers for the paper-figure scripts (argument parsing, data lookup, saving).

Every figure script follows the same pattern::

    ctx = Context.from_args(args, "fig04")       # config + data root + output options
    run = ctx.run("binary256", legacy_time_axis=True)   # raw run dir or bundle
    fig = ...                                    # plotting in the original style
    ctx.save(fig, savefig_kwargs)                # pdf/eps/png + provenance JSON

Paths in ``figures_config.json`` are relative to ``--data-root``.  The default config describes the layout of
the data deposit (``runs/<run>.npz`` bundles, ``cnlm/``, ``eigen/``, ``derived/``);
``figures_config.raw_example.json`` shows the same runs as raw PyUL run directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import numpy as np  # noqa: E402

from Eig_ULDM_packages import trajectories as traj  # noqa: E402

DEFAULT_CONFIG = HERE / "figures_config.json"
VALID_FORMATS = ("pdf", "eps", "png", "svg")
DEFAULT_PNG_DPI = 300  # used when neither --png-dpi nor the original script fixes a PNG resolution


def str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def make_parser(fig_key: str, description: str) -> argparse.ArgumentParser:
    """Argument parser with the options shared by all figure scripts."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default=None,
                        help="root of the deposited data (paths in the config are relative to it); "
                             "defaults to 'data_root' in the config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="figures_config.json")
    parser.add_argument("--out-dir", default="paper_figures_output", help="output directory")
    parser.add_argument("--formats", default=None,
                        help="comma-separated output formats (pdf,eps,png,svg); default from the config")
    parser.add_argument("--basename", default=None, help="output file stem (default: published file name)")
    parser.add_argument("--png-dpi", type=float, default=None, help="PNG resolution (default: the original script's, else 300)")
    parser.add_argument("--prefer", choices=("raw", "bundle"), default="raw",
                        help="use raw run directories or trajectory bundles when both exist")
    parser.add_argument("--axion-mass-ev", type=float, default=None,
                        help="fallback axion mass for runs whose reproducibility.uldm does not record it "
                             "(PYUL_AXION_MASS_EV is used only as a last resort, with a RuntimeWarning)")
    parser.add_argument("--no-provenance", action="store_true", help="do not write <basename>.provenance.json")
    parser.set_defaults(fig_key=fig_key)
    return parser


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Context:
    """Resolved configuration for one figure script invocation."""

    def __init__(self, cfg: dict, data_root: Optional[Path], out_dir: Path, formats: List[str], fig_key: str,
                 prefer: str = "raw", axion_mass_ev: Optional[float] = None, png_dpi: Optional[float] = None,
                 basename: Optional[str] = None, provenance: bool = True, options: Optional[dict] = None):
        self.cfg = cfg
        self.data_root = data_root
        self.out_dir = out_dir
        self.formats = formats
        self.fig_key = fig_key
        self.prefer = prefer
        self.axion_mass_ev = axion_mass_ev
        self.png_dpi = png_dpi
        self.fig_cfg = cfg.get("figures", {}).get(fig_key, {})
        self.basename = basename or self.fig_cfg.get("basename", fig_key)
        self.provenance_enabled = provenance
        self.options = dict(options or {})
        self.inputs: List[dict] = []
        self.written: List[str] = []

    # ------------------------------------------------------------------
    @classmethod
    def from_args(cls, args: argparse.Namespace, fig_key: Optional[str] = None, needs_data: bool = True) -> "Context":
        cfg_path = Path(args.config)
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        root = args.data_root if args.data_root is not None else cfg.get("data_root")
        data_root = Path(root).expanduser().resolve() if root else None
        if needs_data and data_root is None:
            raise SystemExit("error: --data-root is required (the config has data_root = null)")
        if data_root is not None and not data_root.is_dir():
            raise SystemExit(f"error: data root {data_root} is not a directory")
        fmt_text = args.formats if args.formats is not None else ",".join(cfg.get("formats", ["pdf", "eps", "png"]))
        formats = [f.strip().lower() for f in fmt_text.split(",") if f.strip()]
        bad = [f for f in formats if f not in VALID_FORMATS]
        if bad:
            raise SystemExit(f"error: unsupported format(s) {bad}; choose from {VALID_FORMATS}")
        axion = args.axion_mass_ev if args.axion_mass_ev is not None else cfg.get("axion_mass_ev_fallback")
        skip = {"data_root", "config", "out_dir", "formats", "basename", "png_dpi", "prefer",
                "axion_mass_ev", "no_provenance", "fig_key"}
        options = {k: v for k, v in vars(args).items() if k not in skip}
        return cls(cfg, data_root, Path(args.out_dir), formats, fig_key or args.fig_key, prefer=args.prefer,
                   axion_mass_ev=axion, png_dpi=args.png_dpi, basename=args.basename,
                   provenance=not args.no_provenance, options=options)

    # ------------------------------------------------------------------
    def _abs(self, rel: str) -> Path:
        if self.data_root is None:
            raise RuntimeError("no data root configured")
        return self.data_root / Path(rel)

    def run_entry(self, key: str) -> dict:
        try:
            return self.cfg["runs"][key]
        except KeyError:
            raise KeyError(f"run {key!r} is not defined in the config") from None

    def run_path(self, key: str) -> Path:
        """Raw run directory or bundle for ``key`` (honours ``--prefer``)."""
        entry = self.run_entry(key)
        raw = self._abs(entry["path"]) if entry.get("path") else None
        bundle = self._abs(entry["bundle"]) if entry.get("bundle") else None
        order = [raw, bundle] if self.prefer == "raw" else [bundle, raw]
        for cand in order:
            if cand is not None and cand.exists():
                return cand
        raise FileNotFoundError(
            f"run {key!r}: neither {entry.get('path')} nor {entry.get('bundle')} exists under {self.data_root}")

    def run(self, key: str, legacy_time_axis: bool = False) -> "traj.PyULRun":
        path = self.run_path(key)
        run = traj.load_run(path, axion_mass_ev=self.axion_mass_ev, legacy_time_axis=legacy_time_axis)
        self.inputs.append({
            "kind": "run", "key": key,
            "path": self._rel(path),
            "original_run_id": self.run_entry(key).get("original_run_id"),
            "is_bundle": run.is_bundle,
            "pyul_version": run.pyul_version,
            "axion_mass_ev": run.axion_mass_ev,
            "axion_mass_source": run.axion_mass_source,
            "time_axis": run.time_axis_source,
            "n_frames": run.n_frames,
        })
        return run

    def file(self, key: str, hash_file: bool = True) -> Path:
        try:
            entry = self.cfg["files"][key]
        except KeyError:
            raise KeyError(f"file {key!r} is not defined in the config") from None
        path = self._abs(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"file {key!r} not found: {path}")
        rec = {"kind": "file", "key": key, "path": entry["path"], "original_name": entry.get("original_name"),
               "run": entry.get("run")}
        if hash_file:
            digest = sha256_file(path)
            rec["sha256"] = digest
            expected = entry.get("sha256_prefix")
            if expected and not digest.startswith(expected):
                raise ValueError(f"file {key!r} has sha256 {digest[:16]}..., config expects {expected}")
        self.inputs.append(rec)
        return path

    def _rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.data_root)).replace(os.sep, "/")
        except Exception:
            return str(path)

    # ------------------------------------------------------------------
    def output_paths(self, basename: Optional[str] = None) -> Dict[str, Path]:
        stem = basename or self.basename
        return {fmt: self.out_dir / f"{stem}.{fmt}" for fmt in self.formats}

    def png_dpi_for(self, script_default: Optional[float] = None) -> float:
        """--png-dpi, else the original script's PNG dpi, else DEFAULT_PNG_DPI."""
        if self.png_dpi is not None:
            return self.png_dpi
        return script_default if script_default is not None else DEFAULT_PNG_DPI

    def save(self, fig, savefig_kwargs: Optional[dict] = None, *, basename: Optional[str] = None,
             png_dpi: Optional[float] = None, per_format: Optional[Dict[str, dict]] = None,
             close: bool = True) -> List[Path]:
        """Save ``fig`` in every requested format with the original script's savefig options."""
        import matplotlib.pyplot as plt

        self.out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for fmt, path in self.output_paths(basename).items():
            kw = dict(savefig_kwargs or {})
            kw.update((per_format or {}).get(fmt, {}))
            if fmt == "png":
                kw["dpi"] = self.png_dpi_for(png_dpi)
            fig.savefig(path, format=fmt, **kw)
            written.append(path)
        self.written.extend(str(p) for p in written)
        if close:
            plt.close(fig)
        return written

    def write_provenance(self, extra: Optional[dict] = None, basename: Optional[str] = None) -> Optional[Path]:
        if not self.provenance_enabled:
            return None
        import matplotlib

        stem = basename or self.basename
        rec = {
            "figure": self.fig_key,
            "label": self.fig_cfg.get("label"),
            "published": self.fig_cfg.get("published"),
            "script": self.fig_cfg.get("script"),
            "options": {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
                        for k, v in self.options.items()},
            "inputs": self.inputs,
            "outputs": [Path(p).name for p in self.written],
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "matplotlib": matplotlib.__version__, "platform": platform.platform()},
        }
        if extra:
            rec.update(extra)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{stem}.provenance.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, default=_json_default)
            fh.write("\n")
        return path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def load_cnlm(path: Path):
    """Load a c_nlm HDF5 file with the toolkit reader (``c_nlm_integrator.load_cnlm_from_h5``)."""
    from Eig_ULDM_packages.c_nlm_integrator import load_cnlm_from_h5

    t, c, attrs = load_cnlm_from_h5(str(path))
    return np.asarray(t, dtype=np.float64), c, attrs


def soliton_mass_code(ctx: Context, cnlm_file_key: str) -> float:
    """|c|^2 normalisation M_code = soliton mass (config 'soliton_mass_msun', 12e6 Msun) in code units.

    The axion mass is taken from the reproducibility.uldm of the run that owns the c_nlm file
    (falling back to --axion-mass-ev); at 1e-21 eV this gives 170.3149822613627, the value the
    original notebooks used (Units.convert_between(12, 'M_solar_masses', '', 'm')).
    """
    entry = ctx.cfg["files"][cnlm_file_key]
    mass_ev = None
    run_key = entry.get("run")
    if run_key:
        run_entry = ctx.cfg["runs"][run_key]
        repro = ctx._abs(run_entry["path"]) / "reproducibility.uldm" if run_entry.get("path") else None
        if repro is not None and repro.is_file():
            with open(repro, "r", encoding="utf-8") as fh:
                mass_ev = traj.axion_mass_from_reproducibility(json.load(fh))
        elif run_entry.get("bundle"):
            bundle = ctx._abs(run_entry["bundle"])
            if bundle.is_file():
                mass_ev = traj.load_bundle(bundle, axion_mass_ev=ctx.axion_mass_ev).axion_mass_ev
    if mass_ev is None:
        if ctx.axion_mass_ev is None:
            raise ValueError(f"cannot determine the axion mass for {cnlm_file_key}; pass --axion-mass-ev")
        mass_ev = ctx.axion_mass_ev
    msun = float(ctx.cfg.get("soliton_mass_msun", 12e6))
    return float(traj.convert_to_code(msun, "solar_masses", "m", mass_ev))
