"""Per-shell a_lm JSON files with the historical kernel (functions3.process_file).

This is the a_lm stage of the original (paper) pipeline: one JSON file per
(snapshot, shell) under ``alm_coefficients/``, later merged by
``alm_utils.create_sh_hdf5`` (Main.ipynb Step 3).  For new work the one-pass
pipeline ``python -m Eig_ULDM_packages.pipeline`` gives the same numbers much
faster and without PBS.

Run it from the PyUL run's ``Outputs`` directory (or pass ``--outputs-dir``)::

    python compute_coeff.py --rs-length-kpc 5.0 --rs-resol 256
    python compute_coeff.py --start 0 --end 500 --workers 14

Main.ipynb Step 1 copies this file to ``compute_coeff<i>.py`` and replaces the
``r_0_values = np.load(...)`` line so that each copy processes one shell
segment ``r0_segment<i>.npy``.

Configuration priority: command-line option > environment variable
(``ULDM_START_INDEX``, ``ULDM_END_INDEX``, ``ULDM_LMAX``, ``ULDM_RS_LENGTH_KPC``,
``ULDM_RS_RESOL``, ``ULDM_FILE_PREFIX``, ``ULDM_NUM_POINTS``,
``ULDM_ANGULAR_GRID``, ``ULDM_CORES_PER_R0``) > the constants below > run
metadata (``../config.uldm`` and ``../reproducibility.uldm``).

Defaults follow the paper: snapshots 0..save_number, l_max = 10,
num_points = 300, file prefix 3WfnRS/P3R_#, legacy angular grid.  The
resampled-box length and resolution must describe the **3WfnRS** cube (not
the simulation box).  They are read from reproducibility.uldm when PyUL_SK
>= 30.50 recorded them; for older runs they must be given explicitly (with
CellCenteredGrid=True, PyUL 30.38-30.40 resampled onto a side of
(L - L/N)*Length_Ratio).
"""
import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from Eig_ULDM_packages.functions3 import process_file

# ---------------- user constants (None = automatic, see module docstring) ----------------
START_INDEX = 0            # first snapshot (paper: 0)
END_INDEX = None           # last snapshot; None -> save_number from ../config.uldm
L_MAX = 10                 # maximum degree for spherical harmonics (paper: 10)
RS_RESOL = None            # 3WfnRS grid resolution; None -> reproducibility.uldm / cube shape
RS_LENGTH_KPC = None       # 3WfnRS cube side in kpc; None -> reproducibility.uldm (PyUL_SK >= 30.50)
FILE_PREFIX = "3WfnRS/P3R_#"  # or "3Wfn/P3D_#" (then the box is the simulation box)
NUM_POINTS = 300           # N of the N x 2N angular grid (paper: 300)
ANGULAR_GRID = "legacy"    # 'legacy' (paper) or 'dh' (Driscoll-Healy)
WORKERS = None             # processes; None -> CPUs (env/affinity) // CORES_PER_R0
CORES_PER_R0 = 4           # CPUs budgeted per process (memory ~1 GB per process at 256^3)


def process_r0_chunk(params):
    """Function to process a single r0 value"""
    file_index, r_0, l_max, grid_spacing_kpc, pool_size, file_prefix, num_points = params[:7]
    angular_grid = params[7] if len(params) > 7 else "legacy"
    return process_file(file_index, r_0, l_max, grid_spacing_kpc, pool_size, file_prefix, num_points,
                        angular_grid=angular_grid)


def process_file_chunk(chunk_params):
    """Function to process file chunks, maintaining original logic"""
    start_idx, end_idx, r_0_values, l_max, grid_spacing_kpc, pool_size, file_prefix, num_points = chunk_params
    for r_0 in r_0_values:
        for file_index in range(start_idx, end_idx + 1):
            process_file(file_index, r_0, l_max, grid_spacing_kpc, pool_size, file_prefix, num_points)


def load_r0_values():
    r_0_values = np.load("r_0_values.npy") # Load r0 values from file
    return r_0_values


def _env(name, cast, default):
    val = os.environ.get(name, "").strip()
    return cast(val) if val else default


def build_parser():
    p = argparse.ArgumentParser(description="Per-shell a_lm JSON files (historical kernel).")
    p.add_argument("--outputs-dir", default=None, help="PyUL Outputs directory (default: current directory)")
    p.add_argument("--start", type=int, default=_env("ULDM_START_INDEX", int, START_INDEX))
    p.add_argument("--end", type=int, default=_env("ULDM_END_INDEX", int, END_INDEX),
                   help="last snapshot (default: save_number from ../config.uldm)")
    p.add_argument("--l-max", type=int, default=_env("ULDM_LMAX", int, L_MAX))
    p.add_argument("--rs-length-kpc", type=float, default=_env("ULDM_RS_LENGTH_KPC", float, RS_LENGTH_KPC),
                   help="side of the 3WfnRS cube in kpc")
    p.add_argument("--rs-resol", type=int, default=_env("ULDM_RS_RESOL", int, RS_RESOL),
                   help="points per axis of the 3WfnRS cube")
    p.add_argument("--file-prefix", default=_env("ULDM_FILE_PREFIX", str, FILE_PREFIX))
    p.add_argument("--num-points", type=int, default=_env("ULDM_NUM_POINTS", int, NUM_POINTS))
    p.add_argument("--angular-grid", choices=("legacy", "dh"), default=_env("ULDM_ANGULAR_GRID", str, ANGULAR_GRID))
    p.add_argument("--r0-file", default=None, help=".npy with shell radii (kpc); default: see load_r0_values()")
    p.add_argument("--workers", type=int, default=WORKERS, help="number of worker processes")
    p.add_argument("--cores-per-r0", type=int, default=_env("ULDM_CORES_PER_R0", int, CORES_PER_R0))
    p.add_argument("--skip-existing", action="store_true",
                   help="skip shells whose JSON file already exists (only if the tree's kernel_params sidecar "
                        "matches the current dx, RS resolution, num_points, angular grid and l_max)")
    p.add_argument("--no-params-check", action="store_true",
                   help="do not compare with the kernel_params sidecar of an existing JSON tree (and rewrite it "
                        "with the current parameters); --skip-existing then trusts every existing file")
    return p


PARAMS_KEYS = ("grid_spacing_kpc", "rs_resol", "num_points", "angular_grid", "l_max")


def params_sidecar_path(file_prefix, l_max):
    """``alm_coefficients/<folder>/kernel_params_<prefix name>lmax_<l_max>.json`` (one per JSON tree)."""
    from Eig_ULDM_packages._compat import prefix_folder
    name = os.path.basename(str(file_prefix).replace("\\", "/"))
    return os.path.join("alm_coefficients", prefix_folder(file_prefix), f"kernel_params_{name}lmax_{l_max}.json")


def params_mismatch(recorded, current):
    """Keys of PARAMS_KEYS whose recorded value differs from the current one (floats: 1e-12 relative)."""
    import math
    bad = []
    for key in PARAMS_KEYS:
        a, b = recorded.get(key), current.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            if not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0):
                bad.append(key)
        elif a != b:
            bad.append(key)
    return bad


def write_params_sidecar(path, params):
    from Eig_ULDM_packages import __version__
    rec = dict(params, toolkit_version=__version__, written=datetime.now().isoformat(timespec="seconds"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)


def resolve_geometry(args, first_cube_shape):
    """Return (grid_spacing_kpc, rs_resol, source)."""
    resol = args.rs_resol or int(first_cube_shape[0])
    if args.rs_resol and int(first_cube_shape[0]) != int(args.rs_resol):
        raise SystemExit(f"error: --rs-resol {args.rs_resol} but the snapshots are {first_cube_shape}")
    if args.rs_length_kpc is not None:
        return args.rs_length_kpc / resol, resol, "explicit"
    try:
        from Eig_ULDM_packages.pyul_run import PyULRun, rs_geometry
        run = PyULRun.open(os.getcwd())
        geom = rs_geometry(run, units="kpc", cube_resol=resol)
        return geom.side / geom.resol, geom.resol, geom.source
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: cannot determine the 3WfnRS box length ({exc}). "
                         "Pass --rs-length-kpc (and --rs-resol) or set RS_LENGTH_KPC in this script.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.outputs_dir:
        os.chdir(args.outputs_dir)
    r_0_values = np.load(args.r0_file) if args.r0_file else load_r0_values()
    end_index = args.end
    if end_index is None:
        try:
            from Eig_ULDM_packages.pyul_run import PyULRun
            end_index = PyULRun.open(os.getcwd()).save_number
        except (FileNotFoundError, ValueError):
            raise SystemExit("error: --end not given and ../config.uldm not found")
    start_index = args.start
    first_file = f"{args.file_prefix}{start_index:03d}.npy"
    if not os.path.exists(first_file):
        raise SystemExit(f"error: {first_file} not found (cwd={os.getcwd()})")
    grid_spacing_kpc, rs_resol, geo_src = resolve_geometry(args, np.load(first_file, mmap_mode="r").shape)

    from Eig_ULDM_packages._compat import default_workers
    if args.workers:
        n_proc = int(args.workers)
    else:
        n_proc = max(1, default_workers() // max(1, args.cores_per_r0))
    print(f"compute_coeff: snapshots {start_index}..{end_index}, {len(r_0_values)} shells, l_max={args.l_max}, "
          f"num_points={args.num_points}, grid={args.angular_grid}, dx={grid_spacing_kpc:.10g} kpc "
          f"(RS resol {rs_resol}, {geo_src}), {n_proc} processes", flush=True)

    # The JSON files record only (source_file, r_0, coeffs).  A sidecar per JSON tree records the kernel
    # parameters, so that a rerun with other parameters cannot silently reuse (--skip-existing) or partly
    # overwrite files made with different ones.
    params = {"grid_spacing_kpc": float(grid_spacing_kpc), "rs_resol": int(rs_resol),
              "num_points": int(args.num_points), "angular_grid": str(args.angular_grid), "l_max": int(args.l_max)}
    sidecar = params_sidecar_path(args.file_prefix, args.l_max)
    recorded = None
    if os.path.exists(sidecar):
        with open(sidecar, "r") as fh:
            recorded = json.load(fh)
    if recorded is not None and not args.no_params_check:
        bad = params_mismatch(recorded, params)
        if bad:
            diff = ", ".join(f"{k}: existing {recorded.get(k)!r} vs now {params[k]!r}" for k in bad)
            raise SystemExit(f"error: the a_lm JSON tree of {sidecar} was computed with different kernel "
                             f"parameters ({diff}). Use another working directory, remove that tree, or pass "
                             "--no-params-check if you really want to mix or overwrite it.")

    from Eig_ULDM_packages.functions3 import format_file_index
    tasks = []
    n_skipped = 0
    for file_index in range(start_index, end_index + 1):
        for r_0 in r_0_values:
            if args.skip_existing:
                name = os.path.splitext(os.path.basename(f"{args.file_prefix}{format_file_index(file_index)}.npy"))[0]
                from Eig_ULDM_packages._compat import prefix_folder
                js = os.path.join("alm_coefficients", prefix_folder(args.file_prefix), f"r0_{r_0}",
                                  f"lmax_{args.l_max}", f"alm_coefficients_{name}_r0_{r_0}.json")
                if os.path.exists(js):
                    n_skipped += 1
                    continue
            tasks.append((file_index, r_0, args.l_max, grid_spacing_kpc, args.cores_per_r0, args.file_prefix,
                          args.num_points, args.angular_grid))
    if n_skipped and recorded is None and not args.no_params_check:
        raise SystemExit(f"error: --skip-existing would reuse {n_skipped} existing JSON files, but {sidecar} "
                         "(the record of their kernel parameters) does not exist, so they cannot be checked "
                         "(files made before this check existed). Pass --no-params-check to reuse them anyway "
                         "if you know they were made with the same dx, num_points, angular grid and l_max.")
    if recorded is None or args.no_params_check:
        write_params_sidecar(sidecar, params)
    failures = []
    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        futures = {executor.submit(process_r0_chunk, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="a_lm shells", mininterval=10):
            try:
                fut.result()
            except Exception as exc:  # process_file already logged it to error_log.txt
                t = futures[fut]
                failures.append({"snapshot": int(t[0]), "r0": float(t[1]), "error": repr(exc)})
    summary = {"finished": datetime.now().isoformat(timespec="seconds"), "tasks": len(tasks),
               "failures": len(failures), "first_failures": failures[:20], "start": start_index,
               "end": end_index, "n_shells": int(len(r_0_values)), "l_max": args.l_max,
               "num_points": args.num_points, "angular_grid": args.angular_grid,
               "grid_spacing_kpc": grid_spacing_kpc, "rs_resol": rs_resol, "geometry_source": geo_src,
               "skipped_existing": n_skipped, "params_sidecar": sidecar}
    with open(f"compute_coeff_summary_{start_index}-{end_index}_{os.getpid()}.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    if failures:
        print(f"compute_coeff: {len(failures)} of {len(tasks)} shells FAILED (see error_log.txt)", file=sys.stderr)
        return 1
    print(f"compute_coeff: all {len(tasks)} shells done")
    # Optional animation (needs the 'anim' extra), see functions3.create_animation:
    #create_animation(start_index, end_index, grid_spacing_kpc, r_0_values, l_max,
                    #pool_size, file_prefix, plot_num_points, duration, save_number)
    return 0


if __name__ == "__main__":
    sys.exit(main())
