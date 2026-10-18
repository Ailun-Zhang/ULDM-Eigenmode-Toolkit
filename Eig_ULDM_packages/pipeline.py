"""Local (no PBS) a_lm -> c_nlm pipeline for one PyUL run directory.

Usage::

    python -m Eig_ULDM_packages.pipeline RUN_DIR --build-eigen [options]
    uldm-cnlm RUN_DIR --eigen-file eigs.h5 [options]          # console script

``RUN_DIR`` is a PyUL run directory (the folder with ``config.uldm``) or its
``Outputs`` folder.  The pipeline

1. reads the run metadata (duration, save number, axion mass, soliton mass,
   and -- for PyUL_SK >= 30.50 -- the resampled-box geometry);
2. optionally builds the eigenfunction file with the paper defaults
   (soliton mass from config.uldm, ``n_grid=5120`` including r=0,
   ``lmax=10``, ``n_eig=25``, ``r_max`` = half the resampled box);
3. computes a_lm(r) for every requested snapshot
   ``Outputs/3WfnRS/P3R_#NNN.npy`` with the one-pass kernel
   (:mod:`Eig_ULDM_packages.alm_kernel`, bitwise identical to
   ``functions3.process_file``), writing one checkpoint per snapshot so an
   interrupted run resumes where it stopped;
4. writes ``c_nlm.h5`` (layout of ``save_cnlm_to_h5``) and optionally the a_lm
   HDF5 (layout of ``create_sh_hdf5``), computed exactly as the historical
   chain ``process_file -> create_sh_hdf5 -> load_a_lm -> compute_cnlm``.

Runs made with PyUL's inline decomposition (``evolve(Cnlm=True)``, see
:mod:`Eig_ULDM_packages.cnlm_inline`) leave per-snapshot checkpoints in
``Outputs/Cnlm/alm_checkpoints``.  ``uldm-cnlm RUN_DIR --from-inline`` takes
every parameter from ``Outputs/Cnlm/cnlm_inline.json`` and assembles (or, if
the run died, completes) ``c_nlm.h5`` from those checkpoints, computing a
snapshot from its ``3WfnRS`` file only when it has no checkpoint.  Inline
checkpoints need no snapshot file.

The time axis is the real PyUL save times ``Outputs/snapshot_times_Myr.npy``
(PyUL_SK >= 30.50), else ``linspace(0, duration, save_number+1)[snapshots]``.

Defaults reproduce the paper's analysis: 257 shells for a 256^3 resampled
cube (``n_shells = resolR + 1``) from ``r[0] = 1e-5`` to half the box,
``l_max = 10`` (the eigen file's lmax), ``num_points = 300``, the legacy
angular grid, and the c_nlm time axis in Myr described above.  Existing final outputs are never
overwritten without ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import __version__
from ._compat import default_workers
from .pyul_run import PyULRun, rs_geometry

INLINE_PARAMS_FILE = "cnlm_inline.json"
INLINE_DIRNAME = "Cnlm"

__all__ = ["main", "build_parser", "parse_snapshot_spec", "build_eigen_file", "run_pipeline"]

PAPER_EIGEN_DEFAULTS = dict(n_grid=5120, lmax=10, n_eig=25, maxiter=2_000_000, tol=1e-10)


def _log(msg: str) -> None:
    print(f"[uldm-cnlm] {msg}", flush=True)


def parse_snapshot_spec(spec: Optional[str], save_number: int) -> List[int]:
    """'0:500' (inclusive), '0:500:10', '3,7,9' or None (= 0..save_number)."""
    if save_number < 0:
        raise ValueError(f"save number {save_number} < 0: pass the effective save number "
                         "(PyULRun.effective_save_number) for runs with Save Options Number = -1")
    if spec is None or str(spec).strip() in ("", "all"):
        return list(range(0, save_number + 1))
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = part.split(":")
            if len(bits) not in (2, 3):
                raise ValueError(f"bad snapshot range {part!r}")
            a = int(bits[0]) if bits[0] else 0
            b = int(bits[1]) if bits[1] else save_number
            s = int(bits[2]) if len(bits) == 3 and bits[2] else 1
            out.extend(range(a, b + 1, s))
        else:
            out.append(int(part))
    out = sorted(set(out))
    if not out:
        raise ValueError("empty snapshot selection")
    if out[0] < 0 or out[-1] > save_number:
        raise ValueError(f"snapshots must lie in 0..{save_number} (save number of this run)")
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_eigen_file(
    path: str,
    *,
    soliton_mass: float,
    mass_units: str,
    rmax: float,
    length_units: str,
    n_grid: int = 5120,
    lmax: int = 10,
    n_eig: int = 25,
    maxiter: int = 2_000_000,
    tol: float = 1e-10,
    outer_boundary: str = "legacy",
    overwrite: bool = False,
    mass0: Optional[float] = None,
) -> str:
    """Build the eigenfunction HDF5 exactly as ``f_nl_python.ipynb`` (Steps 1-5).

    ``r_grid = linspace(0, rmax_code, n_grid)`` (includes r = 0), potential from
    the bundled ``soliton.h5`` scaled to ``soliton_mass`` with
    ``compute_potential(zero_at_rmax=False)``, ``make_all_eigs(..., maxiter,
    tol)``.  Uses the axion mass currently set in :mod:`units`.  ``mass0``
    overrides the profile's rounded ``mass0 = 3.883`` attribute (``None``:
    keep it, as for all published eigenfunctions).
    """
    from importlib import resources
    from . import units
    from .uldm_eig1 import (load_soliton_file, soliton_density_function, compute_potential,
                            make_all_eigs, save_eigs_to_h5)

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists (use --force to rebuild)")
    soliton_path = resources.files("Eig_ULDM_packages") / "soliton.h5"
    sol = load_soliton_file(str(soliton_path), mass0=mass0)
    mass = units.convert_between(soliton_mass, mass_units, '', 'm')
    rmax_code = units.convert_between(rmax, length_units, '', 'l')
    r_grid = np.linspace(0, rmax_code, int(n_grid))
    rho_func = soliton_density_function(mass, sol)
    phi = compute_potential(r_grid, rho_func, zero_at_rmax=False)
    eigs = make_all_eigs(r_grid, phi, int(lmax), int(n_eig), maxiter=int(maxiter), tol=float(tol),
                         outer_boundary=outer_boundary)
    meta = {
        "soliton_mass": float(soliton_mass), "soliton_mass_units": mass_units,
        "soliton_mass_code": float(mass), "rmax": float(rmax), "rmax_units": length_units,
        "rmax_code": float(rmax_code), "n_grid": int(n_grid), "lmax": int(lmax), "n_eig": int(n_eig),
        "includes_r0": True, "maxiter": int(maxiter), "tol": float(tol),
        "outer_boundary": outer_boundary, "zero_at_rmax": False,
        "axion_mass_ev": float(units.get_axion_mass_ev()),
        "soliton_profile": "Eig_ULDM_packages/soliton.h5",
        "soliton_mass0": "file attribute (3.883)" if mass0 is None else float(mass0),
    }
    tmp = path + ".tmp"
    save_eigs_to_h5(eigs, tmp, metadata=meta)
    os.replace(tmp, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="uldm-cnlm",
        description="Compute a_lm(r,t) and c_nlm(t) for a PyUL run locally (no PBS). "
                    "Defaults reproduce the Stone Skipping paper analysis.",
    )
    p.add_argument("run_dir", help="PyUL run directory (with config.uldm) or its Outputs folder")
    g = p.add_argument_group("input")
    g.add_argument("--input-prefix", default="3WfnRS/P3R_#",
                   help="snapshot prefix relative to Outputs (default 3WfnRS/P3R_#)")
    g.add_argument("--snapshots", default=None,
                   help="'a:b' (inclusive), 'a:b:step', 'i,j,k' or 'all' (default: 0..save_number)")
    g.add_argument("--skip-missing", action="store_true",
                   help="drop snapshots whose file is missing instead of stopping")
    g.add_argument("--axion-mass-ev", type=float, default=None,
                   help="axion mass in eV for runs that do not record it (reproducibility.uldm wins; "
                        "PYUL_AXION_MASS_EV is the last resort)")
    g = p.add_argument_group("resampled-box geometry (auto from reproducibility.uldm for PyUL_SK >= 30.50)")
    g.add_argument("--rs-side", type=float, default=None, help="side of the resampled cube (length units)")
    g.add_argument("--rs-resol", type=int, default=None, help="points per axis of the resampled cube")
    g.add_argument("--rs-dx", type=float, default=None, help="grid spacing of the resampled cube")
    g.add_argument("--length-units", default=None, help="length units of --rs-*/shells (default: run's, kpc)")
    g = p.add_argument_group("a_lm kernel")
    g.add_argument("--n-shells", type=int, default=None, help="number of shells (default resolR+1)")
    g.add_argument("--r-max", type=float, default=None, help="outer shell radius (default half the RS box)")
    g.add_argument("--r0", type=float, default=None, help="radius used for the first shell (default 1e-5)")
    g.add_argument("--shells-file", default=None, help=".npy file with explicit shell radii (length units)")
    g.add_argument("--l-max", type=int, default=None, help="max degree (default: eigen file lmax)")
    g.add_argument("--num-points", type=int, default=None, help="N of the N x 2N angular grid (default 300)")
    g.add_argument("--angular-grid", choices=("legacy", "dh"), default=None,
                   help="sphere sampling: legacy (paper, default) or dh (Driscoll-Healy)")
    g.add_argument("--workers", type=int, default=None,
                   help="threads (default: ULDM_NUM_WORKERS/PBS_NCPUS/SLURM_CPUS_PER_TASK/OMP_NUM_THREADS or CPU affinity)")
    g = p.add_argument_group("eigenfunctions")
    g.add_argument("--eigen-file", default=None, help="existing eigenfunction HDF5 (ell/<l>/{r,E,f})")
    g.add_argument("--build-eigen", action="store_true", help="build the eigen file with the paper defaults")
    g.add_argument("--soliton-mass", type=float, default=None, help="soliton mass (default from config.uldm)")
    g.add_argument("--soliton-mass-units", default=None, help="units of --soliton-mass (default config's)")
    g.add_argument("--eigen-rmax", type=float, default=None, help="eigen grid r_max (default = shell r_max)")
    g.add_argument("--eigen-n-grid", type=int, default=PAPER_EIGEN_DEFAULTS["n_grid"])
    g.add_argument("--eigen-lmax", type=int, default=PAPER_EIGEN_DEFAULTS["lmax"])
    g.add_argument("--eigen-n-eig", type=int, default=PAPER_EIGEN_DEFAULTS["n_eig"])
    g.add_argument("--eigen-maxiter", type=int, default=PAPER_EIGEN_DEFAULTS["maxiter"])
    g.add_argument("--eigen-outer-boundary", choices=("legacy", "rmax"), default="legacy")
    g.add_argument("--soliton-mass0", type=float, default=None,
                   help="override soliton.h5's rounded mass0=3.883 (e.g. 3.8827652755822006); default: keep")
    g = p.add_argument_group("output")
    g.add_argument("--out-dir", default=None, help="output folder (default <run>/Outputs/cnlm_pipeline)")
    g.add_argument("--cnlm-out", default=None, help="c_nlm file name or path (default c_nlm.h5)")
    g.add_argument("--alm-out", default=None, help="also write the a_lm HDF5 (create_sh_hdf5 layout) here")
    g.add_argument("--no-cnlm", action="store_true", help="only compute a_lm (checkpoints / --alm-out)")
    g.add_argument("--force", action="store_true",
                   help="recompute checkpoints and overwrite existing outputs")
    g.add_argument("--dry-run", action="store_true", help="resolve all parameters, print them and exit")
    g.add_argument("--from-inline", action="store_true",
                   help="use the parameters and checkpoints of PyUL's inline decomposition "
                        "(evolve(Cnlm=True); Outputs/Cnlm/cnlm_inline.json) to assemble or complete c_nlm.h5")
    return p


def _apply_inline_settings(args: argparse.Namespace, run: PyULRun) -> Dict[str, Any]:
    """Fill unset options from ``<out_dir or Outputs/Cnlm>/cnlm_inline.json`` (``--from-inline``)."""
    cnlm_dir = os.path.abspath(args.out_dir or os.path.join(run.outputs_dir, INLINE_DIRNAME))
    path = os.path.join(cnlm_dir, INLINE_PARAMS_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"error: --from-inline: {path} not found (was the run made with evolve(Cnlm=True)?)")
    with open(path, "r", encoding="utf-8") as fh:
        rec = json.load(fh)
    k = rec["kernel"]
    args.out_dir = cnlm_dir
    if args.n_shells is None and args.shells_file is None:
        args.n_shells = int(k["n_shells"])
    if args.r_max is None and args.shells_file is None:
        args.r_max = float(k["r_max"])
    if args.r0 is None:
        args.r0 = float(k["r0"])
    if args.l_max is None:
        args.l_max = int(k["l_max"])
    if args.num_points is None:
        args.num_points = int(k["num_points"])
    if args.angular_grid is None:
        args.angular_grid = str(k["angular_grid"])
    rec["_eigen_from_record"] = False
    if args.eigen_file is None and not args.build_eigen and not args.no_cnlm:
        eig = rec["eigen"]["file"]
        alt = os.path.join(cnlm_dir, os.path.basename(eig))
        args.eigen_file = eig if os.path.exists(eig) else alt
        rec["_eigen_from_record"] = True
    outs = rec["outputs"]
    if args.cnlm_out is None:
        rel = outs.get("cnlm_relative_to_outputs")
        args.cnlm_out = os.path.join(run.outputs_dir, rel) if rel else outs["cnlm"]
    if args.alm_out is None and outs.get("alm"):
        rel = outs.get("alm_relative_to_outputs")
        args.alm_out = os.path.join(run.outputs_dir, rel) if rel else outs["alm"]
    return rec


def _resolve_axion_mass(run: PyULRun, cli_value: Optional[float]):
    """Toolkit policy (:func:`.pyul_run.resolve_axion_mass`): reproducibility.uldm, then
    ``--axion-mass-ev``, then ``PYUL_AXION_MASS_EV`` as a last resort (with a warning)."""
    try:
        value, source = run.resolve_axion_mass(cli_value)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    return value, ("command line" if source == "argument" else source)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    from . import units
    from .alm_kernel import AlmKernel, lm_list
    from .cnlm_snapshot import (CnlmProjector, alm_counts_roundtrip, legacy_alm_radial_grid,
                                stack_cnlm_snapshots, write_alm_h5, write_cnlm_h5, INLINE_SOURCE_ID,
                                kernel_fingerprint, checkpoint_path, write_alm_checkpoint,
                                read_alm_checkpoint)
    from .c_nlm_integrator import load_eigenfunctions, check_rmax_consistency
    from ._h5meta import provenance, attrs_to_dict

    t_start = time.time()
    try:
        run = PyULRun.open(args.run_dir)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}")
    inline_rec = _apply_inline_settings(args, run) if getattr(args, "from_inline", False) else None
    for name, default in (("r0", 1e-5), ("num_points", 300), ("angular_grid", "legacy"), ("cnlm_out", "c_nlm.h5")):
        if getattr(args, name, None) is None:
            setattr(args, name, default)
    out_dir = os.path.abspath(args.out_dir or os.path.join(run.outputs_dir, "cnlm_pipeline"))
    ckpt_dir = os.path.join(out_dir, "alm_checkpoints")
    axion_ev, axion_src = _resolve_axion_mass(run, args.axion_mass_ev)
    units.set_axion_mass(axion_ev)
    length_units = args.length_units or run.length_units

    # ---------------- snapshots ----------------
    def _inline_ckpt(k: int) -> bool:
        ck = checkpoint_path(ckpt_dir, k)
        if not os.path.exists(ck):
            return False
        try:
            with np.load(ck) as z:
                return str(z["source_id"]) == INLINE_SOURCE_ID
        except Exception:  # unreadable checkpoint: treated as absent here, refused later
            return False

    ckpt_indices = set()
    if os.path.isdir(ckpt_dir):
        for name in os.listdir(ckpt_dir):
            if name.startswith("ALM_#") and name.endswith(".npz") and name[5:-4].isdigit():
                ckpt_indices.add(int(name[5:-4]))
    # Save Options Number = -1 (save every step) is not kept in config.uldm: count the saves that exist
    try:
        save_number = run.effective_save_number(args.input_prefix,
                                                ckpt_indices if inline_rec is not None else ())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    if inline_rec is not None and args.snapshots is None:
        cand = set(run.available_snapshots(args.input_prefix)) | ckpt_indices
        indices = sorted(k for k in cand if 0 <= k <= save_number)
        if not indices:
            raise SystemExit(f"error: no inline checkpoints in {ckpt_dir} and no snapshot files")
    else:
        try:
            indices = parse_snapshot_spec(args.snapshots, save_number)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")
    paths = {k: run.snapshot_path(args.input_prefix, k) for k in indices}
    inline_ok = {k for k in indices if _inline_ckpt(k)}
    missing = [k for k, pth in paths.items() if not os.path.exists(pth) and k not in inline_ok]
    if missing:
        if not args.skip_missing:
            hint = ""
            if run.save_format != "npy":
                hint = f" (this run saved format {run.save_format!r}; the pipeline reads .npy only)"
            raise SystemExit(f"error: {len(missing)} snapshot files missing, e.g. {paths[missing[0]]}{hint}. "
                             "Run PyUL with CenterCalc=True and the 3WfnRS save flag, choose --snapshots, "
                             "or pass --skip-missing.")
        indices = [k for k in indices if k not in missing]
        if not indices:
            raise SystemExit("error: no snapshot files found")
        _log(f"skipping {len(missing)} missing snapshots: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    try:
        times, time_source = run.snapshot_times_myr(indices, return_source=True)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    with_file = [k for k in indices if os.path.exists(paths[k])]
    cube_n: Optional[int] = None  # only inline checkpoints: the resolution comes from the run's RS record
    if with_file:
        first = np.load(paths[with_file[0]], mmap_mode="r")
        if first.ndim != 3 or len(set(first.shape)) != 1:
            raise SystemExit(f"error: {paths[with_file[0]]} is not a cubic 3-D array (shape {first.shape})")
        cube_n = int(first.shape[0])
        del first

    # ---------------- geometry and shells ----------------
    try:
        geom = rs_geometry(run, rs_side=args.rs_side, rs_resol=args.rs_resol, rs_dx=args.rs_dx,
                           units=length_units, cube_resol=cube_n)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    if cube_n is None:
        cube_n = int(geom.resol)
    if geom.resol != cube_n:
        raise SystemExit(f"error: RS resolution {geom.resol} does not match the cubes ({cube_n}^3)")
    custom_shells = args.shells_file is not None
    if custom_shells:
        r_values = np.asarray(np.load(args.shells_file), dtype=np.float64).ravel()
        r_max = float(r_values[-1])
    else:
        n_r = int(args.n_shells) if args.n_shells else cube_n + 1
        r_max = float(args.r_max) if args.r_max is not None else geom.half_side
        r_values = np.linspace(0, r_max, n_r, endpoint=True)
        r_values[0] = args.r0
    if not np.all(np.diff(r_values) > 0):
        raise SystemExit("error: shell radii must be strictly increasing")

    # ---------------- eigenfunctions ----------------
    os.makedirs(out_dir, exist_ok=True)
    eigen_file = args.eigen_file
    eig_meta: Dict[str, Any] = {}
    if not args.no_cnlm:
        if eigen_file is None and not args.build_eigen:
            raise SystemExit("error: give --eigen-file or --build-eigen (or --no-cnlm)")
        if eigen_file is None:
            masses = run.soliton_masses()
            if args.soliton_mass is not None:
                m_sol = float(args.soliton_mass)
            elif len(masses) == 1 and not run.embedded_solitons:
                m_sol = masses[0]
            else:
                raise SystemExit(f"error: config.uldm has {len(masses)} solitons "
                                 f"(+{len(run.embedded_solitons)} embedded); pass --soliton-mass")
            m_units = args.soliton_mass_units or run.soliton_mass_units
            e_rmax = float(args.eigen_rmax) if args.eigen_rmax is not None else r_max
            eig_meta = dict(soliton_mass=m_sol, mass_units=m_units, rmax=e_rmax, length_units=length_units,
                            n_grid=args.eigen_n_grid, lmax=args.eigen_lmax, n_eig=args.eigen_n_eig,
                            maxiter=args.eigen_maxiter, tol=PAPER_EIGEN_DEFAULTS["tol"],
                            outer_boundary=args.eigen_outer_boundary, mass0=args.soliton_mass0)
            eigen_file = os.path.join(out_dir, f"eigs_mass_{m_sol:g}{m_units}_Eig_{args.eigen_n_eig}_lmax_"
                                               f"{args.eigen_lmax}_nig_{args.eigen_n_grid}.h5")
    # ---------------- l_max ----------------
    eig_lmax = None
    if eig_meta:
        eig_lmax = int(eig_meta["lmax"])
    elif eigen_file is not None:
        if not os.path.exists(eigen_file):
            raise SystemExit(f"error: eigen file not found: {eigen_file}")
        import h5py
        with h5py.File(eigen_file, "r") as fh:
            eig_lmax = max(int(k) for k in fh["ell"].keys())
    l_max = int(args.l_max) if args.l_max is not None else (eig_lmax if eig_lmax is not None else 10)
    if eig_lmax is not None and l_max > eig_lmax:
        warnings.warn(f"l_max={l_max} > eigen lmax={eig_lmax}: c_nlm is only computed for l <= {eig_lmax}",
                      RuntimeWarning)
    workers = default_workers(args.workers)

    fingerprint, fp_hash = kernel_fingerprint(
        toolkit_version=__version__, input_prefix=args.input_prefix, dx=geom.dx, rs_side=geom.side,
        rs_resol=geom.resol, length_units=length_units, r_values=r_values, l_max=l_max,
        num_points=int(args.num_points), angular_grid=args.angular_grid)

    plan = {
        "run_dir": run.run_dir, "out_dir": out_dir, "integrator_version": run.integrator_version,
        "axion_mass_ev": axion_ev, "axion_mass_source": axion_src,
        "snapshots": f"{indices[0]}..{indices[-1]} ({len(indices)} files)",
        "rs_geometry": {"side": geom.side, "resol": geom.resol, "dx": geom.dx, "units": geom.units,
                        "source": geom.source},
        "shells": {"n_r": int(r_values.size), "r_first": float(r_values[0]), "r_max": float(r_values[-1]),
                   "custom": custom_shells},
        "l_max": l_max, "num_points": int(args.num_points), "angular_grid": args.angular_grid,
        "workers": workers, "eigen_file": eigen_file, "build_eigen": bool(eig_meta),
        "eigen_params": eig_meta or None, "fingerprint": fp_hash,
        "from_inline": inline_rec is not None, "inline_checkpoints": len(inline_ok),
    }
    _log("plan: " + json.dumps(plan, indent=1, default=str))
    if args.dry_run:
        return {"plan": plan}

    cnlm_path = None if args.no_cnlm else (args.cnlm_out if os.path.isabs(args.cnlm_out)
                                           else os.path.join(out_dir, args.cnlm_out))
    alm_path = None if args.alm_out is None else (args.alm_out if os.path.isabs(args.alm_out)
                                                   else os.path.join(out_dir, args.alm_out))
    for pth in (cnlm_path, alm_path):
        if pth is not None and os.path.exists(pth) and not args.force:
            raise SystemExit(f"error: {pth} exists; use --force to overwrite or choose another --out-dir")
    if (inline_rec is not None and inline_rec.get("_eigen_from_record") and cnlm_path is not None
            and inline_rec.get("eigen", {}).get("sha256")):
        rec_sha = str(inline_rec["eigen"]["sha256"])
        now_sha = _sha256(eigen_file)
        if now_sha != rec_sha:
            msg = (f"the eigen file {eigen_file} (sha256 {now_sha[:16]}...) is not the one the inline run used "
                   f"(sha256 {rec_sha[:16]}... in {INLINE_PARAMS_FILE}); it was rebuilt or replaced")
            if not args.force:
                raise SystemExit(f"error: {msg}. Pass --eigen-file to use a specific file, or --force to "
                                 "continue with this one.")
            warnings.warn(msg + "; continuing because of --force", RuntimeWarning)

    # ---------------- eigen build (optional) ----------------
    if eig_meta:
        if os.path.exists(eigen_file) and not args.force:
            import h5py
            with h5py.File(eigen_file, "r") as fh:
                have = attrs_to_dict(fh.attrs)
            want = {"soliton_mass": float(eig_meta["soliton_mass"]), "rmax": float(eig_meta["rmax"]),
                    "n_grid": int(eig_meta["n_grid"]), "lmax": int(eig_meta["lmax"]),
                    "n_eig": int(eig_meta["n_eig"]), "axion_mass_ev": float(axion_ev),
                    "outer_boundary": eig_meta["outer_boundary"],
                    "soliton_mass0": ("file attribute (3.883)" if eig_meta["mass0"] is None
                                      else float(eig_meta["mass0"]))}
            bad = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
            if bad:
                raise SystemExit(f"error: existing {eigen_file} was built with different parameters {bad}; "
                                 "use --force or another --out-dir")
            _log(f"reusing eigen file {eigen_file}")
        else:
            t0 = time.time()
            _log(f"building eigen file {eigen_file} (n_grid={eig_meta['n_grid']}, lmax={eig_meta['lmax']}, "
                 f"n_eig={eig_meta['n_eig']}) ...")
            build_eigen_file(eigen_file, soliton_mass=eig_meta["soliton_mass"], mass_units=eig_meta["mass_units"],
                             rmax=eig_meta["rmax"], length_units=length_units, n_grid=eig_meta["n_grid"],
                             lmax=eig_meta["lmax"], n_eig=eig_meta["n_eig"], maxiter=eig_meta["maxiter"],
                             tol=eig_meta["tol"], outer_boundary=eig_meta["outer_boundary"], overwrite=True,
                             mass0=eig_meta["mass0"])
            _log(f"eigen file done in {time.time() - t0:.1f} s")

    # ---------------- a_lm per snapshot (resumable) ----------------
    os.makedirs(ckpt_dir, exist_ok=True)
    params_path = os.path.join(out_dir, "pipeline_params.json")
    if os.path.exists(params_path) and not args.force:
        with open(params_path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("fingerprint") != fingerprint:
            raise SystemExit(f"error: {out_dir} holds checkpoints made with different kernel parameters "
                             f"({params_path}); use --force or another --out-dir")

    def _write_params() -> None:
        with open(params_path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump({"fingerprint": fingerprint, "fingerprint_hash": fp_hash, "toolkit_version": __version__},
                      fh, indent=1)
        os.replace(params_path + ".tmp", params_path)

    # A new directory is claimed now; an existing record is replaced (--force) only after every checkpoint
    # has been validated or computed, so a failed --force call leaves it unchanged.
    if not os.path.exists(params_path):
        _write_params()

    kernel = AlmKernel(r_values, l_max, num_points=args.num_points, angular_grid=args.angular_grid,
                       workers=workers)
    n_lm = len(lm_list(l_max))
    alm_list: List[np.ndarray] = []
    centers: List[np.ndarray] = []
    n_done = 0
    n_inline = 0
    for count, k in enumerate(indices, 1):
        ck = checkpoint_path(ckpt_dir, k)
        has_file = os.path.exists(paths[k])
        src_id = None
        if has_file:
            st = os.stat(paths[k])
            src_id = f"{st.st_size}:{st.st_mtime_ns}"
        if os.path.exists(ck) and (not args.force or not has_file):
            z = read_alm_checkpoint(ck)
            inline = z["source_id"] == INLINE_SOURCE_ID
            ok = (z["fingerprint"] == fp_hash and (inline or z["source_id"] == src_id)
                  and z["alm"].shape == (r_values.size, n_lm))
            if ok:
                alm_list.append(z["alm"])
                centers.append(z["center"])
                n_inline += int(inline)
                continue
            raise SystemExit(f"error: checkpoint {ck} does not match the current parameters or the snapshot "
                             "file changed since it was written; use --force"
                             + (" (inline checkpoints without a 3WfnRS file cannot be recomputed)"
                                if inline and not has_file else ""))
        t0 = time.time()
        psi = np.load(paths[k])
        if psi.shape != (cube_n,) * 3:
            raise SystemExit(f"error: {paths[k]} has shape {psi.shape}, expected {(cube_n,) * 3}")
        alm, c = kernel.compute(psi, geom.dx, center="com", return_center=True)
        del psi
        if not np.all(np.isfinite(alm)):
            raise SystemExit(f"error: non-finite a_lm for snapshot {k} ({paths[k]})")
        write_alm_checkpoint(ck, alm, c, fp_hash, k, src_id)
        alm_list.append(alm)
        centers.append(c)
        n_done += 1
        _log(f"snapshot {k:03d} ({count}/{len(indices)}): {time.time() - t0:.2f} s")
    _log(f"a_lm: {n_done} computed, {len(indices) - n_done} from checkpoints ({n_inline} inline)")
    _write_params()

    r_al = None
    eig_sha = None
    if cnlm_path is not None:
        r_f, f_dict = load_eigenfunctions(eigen_file)
        eig_sha = _sha256(eigen_file)
        if custom_shells:
            r_al = np.asarray(units.convert_between(r_values, length_units, '', 'l'), dtype=np.float64)
            if r_al[-1] > r_f[-1] * (1 + 1e-9):
                raise SystemExit("error: the shells extend beyond the eigen grid r_max")
        else:
            try:
                r_al = legacy_alm_radial_grid(r_max, r_values.size, float(r_f[-1]), r0=args.r0,
                                              length_units=length_units)
            except ValueError as exc:
                raise SystemExit(f"error: {exc}")

    common = provenance(
        "Eig_ULDM_packages.pipeline",
        run_dir=run.run_dir, pyul_integrator_version=run.integrator_version,
        axion_mass_ev=axion_ev, axion_mass_source=axion_src,
        input_prefix=args.input_prefix, snapshot_indices=np.asarray(indices, dtype=np.int64),
        times=times, time_units="Myr", time_axis=time_source,
        r_values=r_values, r_units=length_units,
        rs_side=geom.side, rs_resol=geom.resol, rs_dx=geom.dx, rs_units=geom.units, rs_source=geom.source,
        n_r=int(r_values.size), l_max=l_max, num_points=int(args.num_points), angular_grid=args.angular_grid,
        center="com (|psi|^2-weighted, voxel i at i*dx)",
        kernel="alm_kernel one-pass (bitwise equal to functions3.process_file)",
        sh_normalisation="SHExpandDHC(norm=4, sampling=2, csphase=-1): orthonormal complex Y_lm, Condon-Shortley",
    )
    if r_al is not None:
        common["r_values_code"] = r_al

    result: Dict[str, Any] = {"plan": plan}
    if alm_path is not None:
        masses = run.soliton_masses()
        mass_value = 170.3149822613627
        if len(masses) == 1:
            mass_value = float(units.convert_between(masses[0], run.soliton_mass_units, '', 'm'))
        meta = dict(common, format="uldm-alm-h5", format_version=1,
                    ylm_definition="ylm = a_lm * counts; columns l=0..l_max, m=-l..l",
                    counts=int(args.num_points) * 2 * int(args.num_points), mass_value=mass_value,
                    start_index=int(indices[0]), end_index=int(indices[-1]),
                    expansion_centers=np.asarray(centers).ravel())
        write_alm_h5(alm_path, alm_list, args.num_points, mass_value=mass_value, metadata=meta,
                     overwrite=args.force)
        _log(f"wrote a_lm file {alm_path}")
        result["alm_file"] = alm_path

    if cnlm_path is not None:
        projector = CnlmProjector(r_f, f_dict, r_al, l_max)
        per_snap = [projector.project(alm_counts_roundtrip(a, args.num_points)) for a in alm_list]
        c_dict = stack_cnlm_snapshots(per_snap)
        meta = dict(common, format="uldm-cnlm-h5", format_version=1,
                    eigen_file=os.path.abspath(eigen_file), eigen_file_sha256=eig_sha,
                    radial_interp="interp1d cubic (constant extrapolation) onto the eigen grid, trapezoid",
                    alm_storage_roundtrip=True,
                    normalisation="raw c_nlm (sqrt(code mass)); sum |c|^2 = enclosed mass in code units")
        write_cnlm_h5(cnlm_path, times, c_dict, metadata=meta, overwrite=args.force)
        _log(f"wrote c_nlm file {cnlm_path} ({len(c_dict)} series x {times.size} times)")
        result["cnlm_file"] = cnlm_path

    summary = dict(plan, outputs={k: v for k, v in result.items() if k != "plan"},
                   elapsed_s=round(time.time() - t_start, 2), finished_utc=common["created_utc"])
    with open(os.path.join(out_dir, "pipeline_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    _log(f"done in {time.time() - t_start:.1f} s")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
