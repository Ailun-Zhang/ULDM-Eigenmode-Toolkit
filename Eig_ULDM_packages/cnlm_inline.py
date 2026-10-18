r"""In-simulation c_nlm(t) recorder used by PyUL's ``evolve(Cnlm=True)``.

PyUltraLight2 (PyUL_SK >= 30.50) can decompose the wavefunction while it runs:
at every save (including the initial state) it resamples psi onto the
3WfnRS cube exactly as ``Resample3Box`` does for the ``3WfnRS`` output (the
cube does not have to be written to disk), and hands it to
:class:`InlineCnlmRecorder`.  At the end of the run (also after an AutoStop)
the recorder writes ``Outputs/c_nlm.h5`` in exactly the layout of the toolkit's
``c_nlm.h5`` (root ``t`` in Myr + ``n/<n>/l/<l>/m_<m>`` complex128; provenance
in root attributes only).  This replaces the offline pre-work (copying the
toolkit into ``Outputs``, r0 segments, PBS fan-out of ``compute_coeff.py``,
JSON merge, ``compute_c_nlm.ipynb``).

The numbers are those of the offline chain
``process_file -> create_sh_hdf5 -> load_a_lm -> compute_cnlm`` with the same
parameters: the recorder uses the same one-pass kernel
(:mod:`.alm_kernel`), the a_lm storage round trip and the per-snapshot
projector (:mod:`.cnlm_snapshot`), the RS geometry and run metadata of
:mod:`.pyul_run`, and the checkpoint format of the local pipeline
(:mod:`.pipeline`).  Within one process the result is bitwise identical to
``uldm-cnlm`` run on the saved ``3WfnRS`` files; across processes it agrees to
pyshtools' last-bit FFT noise (about 1e-16 of max|c|).

Life cycle (driven by PyUL)
---------------------------
1. :meth:`CnlmInlineSettings.validate` -- at the start of ``evolve``, before
   any output is touched: dependencies, parameter values, the eigen file (or
   whether one can be built unambiguously), output paths.
2. :meth:`InlineCnlmRecorder.setup` -- after ``reproducibility.uldm`` is written
   and before the first save: axion mass and RS geometry from the run record,
   shells, eigen file (built here when ``eigen_file`` is None/'auto'), r_max
   consistency, kernel and projector; writes ``Outputs/Cnlm/cnlm_inline.json``
   and ``pipeline_params.json``.
3. :meth:`InlineCnlmRecorder.record` -- at every save: a_lm of the cube, a
   crash-safe checkpoint ``Outputs/Cnlm/alm_checkpoints/ALM_#NNN.npz``
   (readable by ``uldm-cnlm``), c_nlm of the snapshot kept in memory.
4. :meth:`InlineCnlmRecorder.finalize` -- writes ``c_nlm.h5`` (and optionally
   the a_lm file in the ``create_sh_hdf5`` layout).

If the run dies, ``uldm-cnlm RUN_DIR --from-inline`` assembles ``c_nlm.h5``
from the checkpoints that exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from . import __version__
from .alm_kernel import ANGULAR_GRIDS, AlmKernel, lm_list
from .cnlm_snapshot import (INLINE_SOURCE_ID, CnlmProjector, alm_counts_roundtrip, checkpoint_path,
                            kernel_fingerprint, legacy_alm_radial_grid, read_alm_checkpoint,
                            stack_cnlm_snapshots, write_alm_checkpoint, write_alm_h5, write_cnlm_h5)
from .pyul_run import PyULRun, rs_geometry

__all__ = [
    "CNLM_DIRNAME",
    "INLINE_PARAMS_FILE",
    "INPUT_PREFIX",
    "EIGEN_PARAM_KEYS",
    "CnlmInlineSettings",
    "InlineCnlmRecorder",
]

CNLM_DIRNAME = "Cnlm"
INLINE_PARAMS_FILE = "cnlm_inline.json"
#: The cube the recorder sees is the one PyUL writes to ``Outputs/3WfnRS/P3R_#NNN.npy``.
INPUT_PREFIX = "3WfnRS/P3R_#"
#: Minimum number of shells sampled at a time (memory bound; see AlmKernel.compute(shell_block=)).
SHELL_BLOCK_MIN = 32
#: Keys accepted in ``eigen_params`` (eigen file built with the paper defaults otherwise).
EIGEN_PARAM_KEYS = ("n_grid", "lmax", "n_eig", "maxiter", "tol", "outer_boundary", "mass0")


def _err(msg: str) -> str:
    return f"PyUL inline c_nlm (Cnlm=True): {msg}"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_writable_dir(d: str) -> None:
    """Raise unless ``d`` (or its nearest existing parent) accepts a new file."""
    probe_dir = os.path.abspath(d)
    while not os.path.isdir(probe_dir):
        parent = os.path.dirname(probe_dir)
        if parent == probe_dir:
            break
        probe_dir = parent
    probe = os.path.join(probe_dir, f".cnlm_write_test_{os.getpid()}")
    try:
        with open(probe, "w") as fh:
            fh.write("x")
        os.remove(probe)
    except OSError as exc:
        raise PermissionError(_err(f"cannot write to {probe_dir} ({exc})")) from None


def _path_inside(path: str, folder: str) -> bool:
    """True if ``path`` (after resolving links) is ``folder`` or lies below it."""
    p = os.path.normcase(os.path.realpath(path))
    d = os.path.normcase(os.path.realpath(folder))
    try:
        return os.path.commonpath([p, d]) == d
    except ValueError:  # different drives on Windows
        return False


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


@dataclass
class CnlmInlineSettings:
    """Validated user settings of the inline decomposition (phase 1).

    Construct with :meth:`validate`.  All paths are absolute.
    """

    run_dir: str
    eigen_file: Optional[str]          # None = build in setup (paper defaults)
    l_max: int = 10
    n_shells: int = 257
    r_max: Optional[float] = None      # run length units; None = half the RS side
    num_points: int = 300
    angular_grid: str = "legacy"
    output: str = ""                   # absolute c_nlm.h5 path
    alm_output: Optional[str] = None   # absolute a_lm path, or None
    threads: Optional[int] = None
    soliton_mass: Optional[float] = None
    soliton_mass_units: Optional[str] = None
    eigen_params: Dict[str, Any] = field(default_factory=dict)
    overwrite: bool = False
    r0: float = 1e-5
    eigen_lmax: Optional[int] = None
    eigen_source: str = ""

    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.run_dir, "Outputs")

    @property
    def cnlm_dir(self) -> str:
        return os.path.join(self.outputs_dir, CNLM_DIRNAME)

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.cnlm_dir, "alm_checkpoints")

    # ------------------------------------------------------------------
    @classmethod
    def validate(
        cls,
        run_dir: str,
        *,
        eigen_file: Optional[str] = None,
        l_max: int = 10,
        n_shells: int = 257,
        r_max: Optional[float] = None,
        num_points: int = 300,
        angular_grid: str = "legacy",
        save_alm: Union[bool, str] = False,
        output: str = "c_nlm.h5",
        threads: Optional[int] = None,
        soliton_mass: Optional[float] = None,
        eigen_params: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        use_init: bool = False,
        outputs_replaced: bool = True,
    ) -> "CnlmInlineSettings":
        """Check everything that can be checked before the simulation starts.

        ``run_dir`` is the PyUL run folder (with ``config.uldm``).  Relative
        ``output`` / ``save_alm`` names are inside ``<run_dir>/Outputs``.
        ``use_init=True`` (a loaded initial wavefunction) makes an automatic
        eigen file ambiguous unless ``soliton_mass`` is given.  Raises
        ``ImportError``, ``ValueError``, ``FileNotFoundError``,
        ``FileExistsError`` or ``PermissionError`` with a message naming the
        ``evolve`` keyword to change.

        ``outputs_replaced=True`` (default; what ``evolve`` does): when the run
        starts, ``<run_dir>/Outputs`` is either new or deleted and recreated
        (OverwriteOutputs=True, Silent=True or answering Y), so a
        ``CnlmEigenFile`` inside it is refused here, before anything is deleted.
        """
        for mod, hint in (("pyshtools", "pip install pyshtools (or conda install -c conda-forge pyshtools)"),
                          ("h5py", "pip install h5py"), ("scipy.ndimage", "pip install scipy")):
            try:
                __import__(mod)
            except Exception as exc:  # ImportError or a broken binary
                raise ImportError(_err(f"{mod} is required but cannot be imported ({exc}); {hint}")) from None

        run_dir = os.path.abspath(os.fspath(run_dir))
        run = PyULRun.open(run_dir)
        if not _is_int(l_max) or l_max < 0:
            raise ValueError(_err(f"CnlmLmax must be a non-negative integer, got {l_max!r}"))
        if not _is_int(num_points) or num_points < 2 or num_points % 2:
            raise ValueError(_err(f"CnlmNumPoints must be an even integer >= 2, got {num_points!r}"))
        if l_max > num_points // 2 - 1:
            raise ValueError(_err(f"CnlmLmax={l_max} needs CnlmNumPoints >= {2 * (l_max + 1)} "
                                  f"(SHExpandDHC: l_max <= N/2 - 1)"))
        if not _is_int(n_shells) or n_shells < 2:
            raise ValueError(_err(f"CnlmNShells must be an integer >= 2, got {n_shells!r}"))
        if r_max is not None:
            r_max = float(r_max)
            if not (np.isfinite(r_max) and r_max > 0):
                raise ValueError(_err(f"CnlmRmax must be positive, got {r_max!r}"))
        if angular_grid not in ANGULAR_GRIDS:
            raise ValueError(_err(f"CnlmAngularGrid must be one of {ANGULAR_GRIDS}, got {angular_grid!r}"))
        if threads is not None and (not _is_int(threads) or threads < 1):
            raise ValueError(_err(f"CnlmThreads must be None or a positive integer, got {threads!r}"))
        eigen_params = dict(eigen_params or {})
        bad = sorted(set(eigen_params) - set(EIGEN_PARAM_KEYS))
        if bad:
            raise ValueError(_err(f"unknown CnlmEigenParams keys {bad}; allowed: {list(EIGEN_PARAM_KEYS)}"))

        # ---------------- eigenfunctions ----------------
        auto = eigen_file is None or (isinstance(eigen_file, str) and eigen_file.strip().lower() == "auto")
        s_units = run.soliton_mass_units
        if auto:
            if eigen_params.get("lmax", l_max) < l_max:
                raise ValueError(_err(f"CnlmEigenParams lmax={eigen_params['lmax']} < CnlmLmax={l_max}"))
            if soliton_mass is None:
                masses = run.soliton_masses()
                n_emb = len(run.embedded_solitons)
                if use_init:
                    raise ValueError(_err(
                        "the eigenfunctions cannot be built automatically for a run that starts from a loaded "
                        "wavefunction (UseInit=True): pass CnlmSolitonMass=<mass in the config's soliton mass "
                        f"units ({s_units})> or CnlmEigenFile=<eigen HDF5>"))
                if len(masses) != 1 or n_emb:
                    raise ValueError(_err(
                        f"the eigenfunctions are built from the run's single soliton, but config.uldm has "
                        f"{len(masses)} standard and {n_emb} embedded solitons: pass CnlmSolitonMass=<mass in "
                        f"{s_units}> or CnlmEigenFile=<eigen HDF5>"))
                soliton_mass = masses[0]
            soliton_mass = float(soliton_mass)
            if not (np.isfinite(soliton_mass) and soliton_mass > 0):
                raise ValueError(_err(f"CnlmSolitonMass must be positive, got {soliton_mass!r}"))
            eigen_lmax = int(eigen_params.get("lmax", l_max))
            eig_path = None
            eigen_source = "auto"
        else:
            eig_path = os.path.abspath(os.fspath(eigen_file))
            if not os.path.isfile(eig_path):
                raise FileNotFoundError(_err(f"CnlmEigenFile not found: {eig_path}"))
            if outputs_replaced and _path_inside(eig_path, os.path.join(run_dir, "Outputs")):
                raise ValueError(_err(
                    f"CnlmEigenFile {eig_path} lies inside {os.path.join(run_dir, 'Outputs')}, which evolve "
                    "deletes and recreates before the run starts (OverwriteOutputs=True, Silent=True or answering "
                    "Y), so the eigen file would be deleted. Copy it outside the Outputs folder (for example into "
                    "the run folder or the save path) and pass that copy as CnlmEigenFile. Nothing was deleted."))
            import h5py
            try:
                with h5py.File(eig_path, "r") as fh:
                    ells = sorted(int(k) for k in fh["ell"].keys())
                    for k in ells:
                        g = fh["ell"][str(k)]
                        if "r" not in g or "f" not in g:
                            raise KeyError(f"ell/{k} lacks r or f")
            except (OSError, KeyError, ValueError) as exc:
                raise ValueError(_err(f"CnlmEigenFile {eig_path} is not an eigenfunction file "
                                      f"(ell/<l>/{{r,E,f}}): {exc}")) from None
            eigen_lmax = max(ells)
            if l_max > eigen_lmax:
                raise ValueError(_err(f"CnlmLmax={l_max} exceeds the eigen file's lmax={eigen_lmax}"))
            eigen_source = "file"

        # ---------------- outputs ----------------
        outputs_dir = os.path.join(run_dir, "Outputs")

        def _resolve(name: str) -> str:
            name = os.path.expanduser(os.fspath(name))
            return os.path.abspath(name if os.path.isabs(name) else os.path.join(outputs_dir, name))

        out_path = _resolve(output)
        if not out_path.endswith((".h5", ".hdf5")):
            raise ValueError(_err(f"CnlmOutput should be an HDF5 file name ending in .h5, got {output!r}"))
        alm_path = None
        if save_alm:
            alm_name = "a_lm.h5" if save_alm is True else save_alm
            if not isinstance(alm_name, str):
                raise ValueError(_err(f"CnlmSaveAlm must be False, True or a file name, got {save_alm!r}"))
            alm_path = _resolve(alm_name)
            if alm_path == out_path:
                raise ValueError(_err("CnlmSaveAlm and CnlmOutput name the same file"))
        for pth in [out_path] + ([alm_path] if alm_path else []):
            if os.path.exists(pth) and not overwrite:
                raise FileExistsError(_err(f"{pth} already exists; pass OverwriteOutputs=True or choose "
                                           "another CnlmOutput / CnlmSaveAlm name"))
            _check_writable_dir(os.path.dirname(pth))
        _check_writable_dir(run_dir)
        return cls(run_dir=run_dir, eigen_file=eig_path, l_max=int(l_max), n_shells=int(n_shells), r_max=r_max,
                   num_points=int(num_points), angular_grid=str(angular_grid), output=out_path,
                   alm_output=alm_path, threads=None if threads is None else int(threads),
                   soliton_mass=soliton_mass if auto else None, soliton_mass_units=s_units if auto else None,
                   eigen_params=eigen_params, overwrite=bool(overwrite), eigen_lmax=eigen_lmax,
                   eigen_source=eigen_source)


class InlineCnlmRecorder:
    """Per-save a_lm / c_nlm recorder (phases 2-4, see the module docstring)."""

    def __init__(self, settings: CnlmInlineSettings, *, default_threads: Optional[int] = None,
                 log=None) -> None:
        self.settings = settings
        self.threads = int(settings.threads or default_threads or 1)
        self._log = log or (lambda msg: None)
        self._indices: List[int] = []
        self._times: List[float] = []
        # c_nlm of each recorded save as one complex128 vector in self._cnlm_keys order (about 48 KB per
        # save at l_max 10, n_eig 25, instead of about 450 KB for a dict of numpy scalars)
        self._cnlm: List[np.ndarray] = []
        self._cnlm_keys: Optional[List[Any]] = None
        self._timing: List[float] = []
        self.ready = False
        self.finalized = False

    # ------------------------------------------------------------------
    def setup(self) -> "InlineCnlmRecorder":
        """Resolve geometry/eigenfunctions and build the kernel (before the first save)."""
        from . import units
        from .c_nlm_integrator import load_eigenfunctions
        from ._h5meta import provenance

        st = self.settings
        t0 = time.time()
        run = PyULRun.open(st.run_dir)
        self.run = run
        try:
            axion_ev, axion_src = run.resolve_axion_mass(None, allow_env=False)
        except ValueError as exc:
            raise ValueError(_err(str(exc))) from None
        units.set_axion_mass(axion_ev)
        self.axion_ev, self.axion_src = axion_ev, axion_src
        try:
            geom = rs_geometry(run)
        except ValueError as exc:
            raise ValueError(_err(f"cannot resolve the resampled-box geometry: {exc}")) from None
        self.geom = geom
        self.length_units = run.length_units
        r_max = float(st.r_max) if st.r_max is not None else geom.half_side
        if r_max > geom.half_side * (1 + 1e-12):
            warnings.warn(_err(f"CnlmRmax={r_max:g} {self.length_units} exceeds half the resampled box "
                               f"({geom.half_side:g}); points outside the cube count as psi = 0"),
                          RuntimeWarning, stacklevel=2)
        self.r_max = r_max
        r_values = np.linspace(0, r_max, st.n_shells, endpoint=True)
        r_values[0] = st.r0
        if not np.all(np.diff(r_values) > 0):
            raise ValueError(_err(f"shell radii are not increasing (r0={st.r0} >= r[1]={r_values[1]}); "
                                  "increase CnlmRmax or reduce CnlmNShells"))
        self.r_values = r_values
        os.makedirs(st.checkpoint_dir, exist_ok=True)

        # ---------------- eigenfunctions ----------------
        from .pipeline import PAPER_EIGEN_DEFAULTS, build_eigen_file
        if st.eigen_file is None:
            ep = dict(PAPER_EIGEN_DEFAULTS)
            ep.update(lmax=st.l_max, outer_boundary="legacy", mass0=None)
            ep.update(st.eigen_params)
            name = (f"eigs_mass_{st.soliton_mass:g}{st.soliton_mass_units}_Eig_{int(ep['n_eig'])}_lmax_"
                    f"{int(ep['lmax'])}_nig_{int(ep['n_grid'])}.h5")
            eig_path = os.path.join(st.cnlm_dir, name)
            te = time.time()
            build_eigen_file(eig_path, soliton_mass=st.soliton_mass, mass_units=st.soliton_mass_units,
                             rmax=r_max, length_units=self.length_units, n_grid=int(ep["n_grid"]),
                             lmax=int(ep["lmax"]), n_eig=int(ep["n_eig"]), maxiter=int(ep["maxiter"]),
                             tol=float(ep["tol"]), outer_boundary=ep["outer_boundary"], overwrite=True,
                             mass0=ep["mass0"])
            self._log(f"built eigen file {eig_path} in {time.time() - te:.1f} s")
            self.eigen_params = ep
        else:
            eig_path = st.eigen_file
            self.eigen_params = None
        self.eigen_file = eig_path
        self.eigen_sha256 = _sha256(eig_path)
        r_f, f_dict = load_eigenfunctions(eig_path)
        eig_lmax = max(l for (l, n) in f_dict)
        if st.l_max > eig_lmax:
            raise ValueError(_err(f"CnlmLmax={st.l_max} exceeds the eigen file's lmax={eig_lmax}"))
        try:
            r_al = legacy_alm_radial_grid(r_max, st.n_shells, float(r_f[-1]), r0=st.r0,
                                          length_units=self.length_units)
        except ValueError as exc:
            raise ValueError(_err(f"the eigen file's r_max does not match the shells "
                                  f"(r_max = {r_max:g} {self.length_units}): {exc}")) from None
        self.r_al = r_al
        self.kernel = AlmKernel(r_values, st.l_max, num_points=st.num_points, angular_grid=st.angular_grid,
                                workers=self.threads)
        # sample/expand SHELL_BLOCK shells at a time: bounded memory, bitwise identical values
        self.shell_block = max(SHELL_BLOCK_MIN, 4 * self.threads)
        self.projector = CnlmProjector(r_f, f_dict, r_al, st.l_max)
        self.n_lm = len(lm_list(st.l_max))
        self.fingerprint, self.fp_hash = kernel_fingerprint(
            toolkit_version=__version__, input_prefix=INPUT_PREFIX, dx=geom.dx, rs_side=geom.side,
            rs_resol=geom.resol, length_units=self.length_units, r_values=r_values, l_max=st.l_max,
            num_points=st.num_points, angular_grid=st.angular_grid)
        params_path = os.path.join(st.cnlm_dir, "pipeline_params.json")
        with open(params_path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump({"fingerprint": self.fingerprint, "fingerprint_hash": self.fp_hash,
                       "toolkit_version": __version__}, fh, indent=1)
        os.replace(params_path + ".tmp", params_path)

        env = run.repro.get("Run Environment", {}) if isinstance(run.repro, dict) else {}
        self.pyul_info = {"pyul_version": run.integrator_version,
                          "pyul_core_sha256": env.get("Core SHA256"),
                          "pyul_core_sha256_lf": env.get("Core SHA256 (LF normalised)"),
                          "pyul_d_version": env.get("D_version")}
        self.record_json = {
            "format": "pyul-inline-cnlm-settings", "format_version": 1,
            "toolkit_version": __version__, "run_dir": st.run_dir, **self.pyul_info,
            "axion_mass_ev": axion_ev, "axion_mass_source": axion_src,
            "rs_geometry": {"side": geom.side, "resol": geom.resol, "dx": geom.dx, "units": geom.units,
                            "source": geom.source, "legacy_resample_box": geom.legacy_resample_box,
                            "cell_centered": geom.cell_centered},
            "kernel": {"n_shells": st.n_shells, "r_max": r_max, "r0": st.r0, "l_max": st.l_max,
                       "num_points": st.num_points, "angular_grid": st.angular_grid, "threads": self.threads,
                       "input_prefix": INPUT_PREFIX, "center": "com"},
            "eigen": {"file": eig_path, "sha256": self.eigen_sha256, "source": st.eigen_source,
                      "built_params": self.eigen_params, "lmax": eig_lmax},
            "outputs": {"cnlm": st.output, "alm": st.alm_output,
                        "cnlm_relative_to_outputs": self._rel(st.output),
                        "alm_relative_to_outputs": self._rel(st.alm_output),
                        "checkpoints": st.checkpoint_dir},
            "fingerprint": self.fingerprint, "fingerprint_hash": self.fp_hash,
            "status": "running", "snapshots_recorded": 0,
            "resume_hint": "uldm-cnlm <run_dir> --from-inline",
        }
        self._write_record()
        self._provenance = provenance
        self.ready = True
        self._log(f"inline c_nlm ready in {time.time() - t0:.1f} s: RS {geom.resol}^3, dx={geom.dx:.6g} "
                  f"{geom.units}, {st.n_shells} shells to {r_max:g} {self.length_units}, l_max={st.l_max}, "
                  f"N={st.num_points} ({st.angular_grid}), {self.threads} threads, eigen {os.path.basename(eig_path)}")
        return self

    def _rel(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        rel = os.path.relpath(path, self.settings.outputs_dir)
        return None if rel.startswith("..") or os.path.isabs(rel) else rel

    def _write_record(self) -> None:
        p = os.path.join(self.settings.cnlm_dir, INLINE_PARAMS_FILE)
        with open(p + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(self.record_json, fh, indent=1, default=str)
        os.replace(p + ".tmp", p)

    # ------------------------------------------------------------------
    def record(self, index: int, psi_rs: np.ndarray, t_myr: float) -> float:
        """Decompose the resampled cube of save ``index`` (time ``t_myr``); returns the seconds spent."""
        if not self.ready:
            raise RuntimeError(_err("record() called before setup()"))
        t0 = time.time()
        psi_rs = np.asarray(psi_rs)
        if psi_rs.shape != (self.geom.resol,) * 3:
            raise ValueError(_err(f"cube shape {psi_rs.shape} != RS record {(self.geom.resol,) * 3}"))
        alm, c = self.kernel.compute(psi_rs, self.geom.dx, center="com", return_center=True,
                                     shell_block=self.shell_block)
        if not np.all(np.isfinite(alm)):
            raise FloatingPointError(_err(f"non-finite a_lm at save {index} (the wavefunction is not finite)"))
        write_alm_checkpoint(checkpoint_path(self.settings.checkpoint_dir, index), alm, c, self.fp_hash,
                             int(index), INLINE_SOURCE_ID, t_myr=float(t_myr))
        snap = self.projector.project(alm_counts_roundtrip(alm, self.settings.num_points))
        if self._cnlm_keys is None:
            self._cnlm_keys = list(snap.keys())
        elif list(snap.keys()) != self._cnlm_keys:  # pragma: no cover - the projector order is fixed
            raise RuntimeError(_err("the c_nlm mode order changed between saves"))
        self._cnlm.append(np.array([snap[k] for k in self._cnlm_keys], dtype=np.complex128))
        self._indices.append(int(index))
        self._times.append(float(t_myr))
        dt = time.time() - t0
        self._timing.append(dt)
        return dt

    @property
    def indices(self) -> List[int]:
        return list(self._indices)

    def finalize(self, times_myr: Optional[Sequence[float]] = None, *, stopped_early: bool = False) -> Dict[str, Any]:
        """Write ``c_nlm.h5`` (and the a_lm file if requested); returns the output paths.

        ``times_myr`` (one value per recorded save, in order) defaults to the times
        passed to :meth:`record`.  PyUL passes the array it writes to
        ``Outputs/snapshot_times_Myr.npy``.
        """
        st = self.settings
        if self.finalized:
            return self.result
        if not self._cnlm:
            raise RuntimeError(_err("no snapshot was recorded"))
        t = np.asarray(self._times if times_myr is None else times_myr, dtype=np.float64).ravel()
        if t.size != len(self._indices):
            raise ValueError(_err(f"{t.size} times for {len(self._indices)} recorded saves"))
        stacked = np.stack(self._cnlm)  # (n_saves, n_modes); same values as stack_cnlm_snapshots
        c_dict = {k: np.ascontiguousarray(stacked[:, j]) for j, k in enumerate(self._cnlm_keys)}
        del stacked
        geom = self.geom
        common = self._provenance(
            "Eig_ULDM_packages.cnlm_inline",
            source="pyul-inline", run_dir=st.run_dir, **{k: v for k, v in self.pyul_info.items() if v is not None},
            pyul_integrator_version=self.run.integrator_version,
            axion_mass_ev=self.axion_ev, axion_mass_source=self.axion_src,
            input_prefix=INPUT_PREFIX + " (in memory, as Resample3Box produces it)",
            snapshot_indices=np.asarray(self._indices, dtype=np.int64),
            times=t, time_units="Myr",
            time_axis="PyUL save times start + k*duration/save_number (Outputs/snapshot_times_Myr.npy)",
            stopped_early=bool(stopped_early),
            r_values=self.r_values, r_units=self.length_units, r_values_code=self.r_al,
            rs_side=geom.side, rs_resol=geom.resol, rs_dx=geom.dx, rs_units=geom.units, rs_source=geom.source,
            rs_legacy_resample_box=geom.legacy_resample_box, rs_cell_centered_grid=geom.cell_centered,
            n_r=int(self.r_values.size), l_max=st.l_max, num_points=st.num_points, angular_grid=st.angular_grid,
            threads=self.threads,
            center="com (|psi|^2-weighted, voxel i at i*dx)",
            kernel="alm_kernel one-pass (bitwise equal to functions3.process_file)",
            sh_normalisation="SHExpandDHC(norm=4, sampling=2, csphase=-1): orthonormal complex Y_lm, Condon-Shortley",
            fingerprint=self.fp_hash,
        )
        result: Dict[str, Any] = {}
        meta = dict(common, format="uldm-cnlm-h5", format_version=1,
                    eigen_file=os.path.abspath(self.eigen_file), eigen_file_sha256=self.eigen_sha256,
                    radial_interp="interp1d cubic (constant extrapolation) onto the eigen grid, trapezoid",
                    alm_storage_roundtrip=True,
                    normalisation="raw c_nlm (sqrt(code mass)); sum |c|^2 = enclosed mass in code units")
        write_cnlm_h5(st.output, t, c_dict, metadata=meta, overwrite=st.overwrite)
        result["cnlm_file"] = st.output
        if st.alm_output:
            from . import units
            alm_list, centers = [], []
            for k in self._indices:
                z = read_alm_checkpoint(checkpoint_path(st.checkpoint_dir, k))
                alm_list.append(z["alm"])
                centers.append(z["center"])
            masses = self.run.soliton_masses()
            mass_value = 170.3149822613627
            if len(masses) == 1:
                mass_value = float(units.convert_between(masses[0], self.run.soliton_mass_units, '', 'm'))
            ameta = dict(common, format="uldm-alm-h5", format_version=1,
                         ylm_definition="ylm = a_lm * counts; columns l=0..l_max, m=-l..l",
                         counts=st.num_points * 2 * st.num_points, mass_value=mass_value,
                         start_index=int(self._indices[0]), end_index=int(self._indices[-1]),
                         expansion_centers=np.asarray(centers).ravel())
            write_alm_h5(st.alm_output, alm_list, st.num_points, mass_value=mass_value, metadata=ameta,
                         overwrite=st.overwrite)
            result["alm_file"] = st.alm_output
        tm = np.asarray(self._timing)
        self.record_json.update(status="finished" + (" (stopped early)" if stopped_early else ""),
                                snapshots_recorded=len(self._indices),
                                snapshot_indices=[int(i) for i in self._indices],
                                seconds_per_save={"mean": float(tm.mean()), "max": float(tm.max()),
                                                  "total": float(tm.sum())})
        self._write_record()
        self.finalized = True
        self.result = result
        self._log(f"wrote {st.output} ({len(c_dict)} series x {t.size} times; "
                  f"{tm.mean():.2f} s per save on average)")
        return result
