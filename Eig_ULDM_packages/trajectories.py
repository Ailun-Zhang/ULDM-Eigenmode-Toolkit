"""Load PyUltraLight (PyUL) run outputs and compute black-hole trajectory observables.

This module reads a PyUL run directory without importing PyUL itself:

* ``config.uldm`` and ``reproducibility.uldm`` (JSON),
* the per-save state vectors ``Outputs/NBody/NTM_#NNN``, ``Outputs/ULDCOM/UCM_#NNN`` and
  ``Outputs/DF/DYF_#NNN`` in any of PyUL's save formats (``npy``, ``npz``, ``hdf5``; the
  pre-2022 flat layout ``Outputs/NTM_#NNN.npy`` is also accepted),
* the flat per-save energy arrays ``Outputs/*list.npy`` and ``Outputs/ULDMass.npy``,
* explicit snapshot times ``Outputs/snapshot_times_<unit>.npy`` when present (written by
  PyUL >= 30.50 and by some post-processing notebooks).

Time axis
---------
PyUL saves frame ``k`` (``k = 0 .. save_number``) at ``t_k = start + k * duration / save_number``.
This is what :func:`pyul_time_axis_myr` returns by default.  Several of the original analysis
notebooks (``plot.ipynb``, ``Test_PyUL_alan_2.ipynb``, ``Test_PyUL_alan_Binary.ipynb``) used
``t_k = k * duration / (save_number + 1)`` instead, which compresses the axis by
``save_number / (save_number + 1)`` (the last frame of a 3000 Myr / 500-save run is drawn at
2994.01 Myr).  Pass ``legacy_time_axis=True`` to reproduce that convention exactly; it exists only
to regenerate published figures that were made with it.

Units
-----
Positions are converted from PyUL code length to parsec with the axion mass recorded in the run's
``reproducibility.uldm`` (``"Axion Mass"`` in kg and ``"m22"``).  If the file does not record it,
the caller should pass ``axion_mass_ev`` explicitly; the ``PYUL_AXION_MASS_EV`` environment variable
is used only as a last resort, with a ``RuntimeWarning``.  Run reading, units and this policy are
implemented once, in :mod:`Eig_ULDM_packages.pyul_run`; this module adds the frames and bundles.  At ``m = 1e-21 eV`` one code length is 12133.0958 pc
(:func:`verify_length_conversion`).

Trajectory bundles
------------------
:func:`export_bundle` writes one compressed ``.npz`` per run holding the frame indices, NBody,
ULDCOM and DF state vectors, energies, ULDM mass, particle masses, explicit snapshot times (if any)
and the verbatim text of ``config.uldm`` and ``reproducibility.uldm``.  :func:`load_run` accepts
either a raw run directory or such a bundle and returns the same :class:`PyULRun` object, so plotting
code does not need to know which one it was given.  Bundles contain no pickled objects.

Only ``numpy`` is required; ``h5py`` is imported lazily when an ``hdf5``-format run is read.
Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "REFERENCE_AXION_MASS_EV",
    "REFERENCE_PC_PER_CODE_LENGTH",
    "BUNDLE_FORMAT",
    "BUNDLE_FORMAT_VERSION",
    "PyULRun",
    "code_units",
    "pc_per_code_length",
    "myr_per_code_time",
    "msun_per_code_mass",
    "kms_per_code_velocity",
    "convert_to_code",
    "convert_from_code",
    "verify_length_conversion",
    "axion_mass_from_reproducibility",
    "resolve_axion_mass",
    "find_run_dir",
    "pyul_time_axis_myr",
    "load_frames",
    "load_run",
    "load_bundle",
    "export_bundle",
    "distance_from_origin_pc",
    "distance_from_uldm_com_pc",
]

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# PyUL unit system, axion-mass policy, run directory and time axis: single
# implementation in Eig_ULDM_packages.pyul_run (re-exported here unchanged).
# ---------------------------------------------------------------------------
from . import pyul_run as _pr  # noqa: E402
from .pyul_run import (  # noqa: E402,F401  (re-exported public names)
    HBAR, PARSEC, LIGHT_YEAR, SOLAR_MASS, G_NEWTON, OMEGA_M0, H_0, EV_KG, SECONDS_PER_YEAR,
    REFERENCE_AXION_MASS_EV, code_units, pc_per_code_length, myr_per_code_time, msun_per_code_mass,
    kms_per_code_velocity, convert_to_code, convert_from_code, axion_mass_from_reproducibility,
    pyul_time_axis_myr, _check_positive, _TIME_SECONDS,
)

#: Code length in pc at 1e-21 eV, as quoted in the paper / used by every published figure.
REFERENCE_PC_PER_CODE_LENGTH = 12133.0958

BUNDLE_FORMAT = "pyul-trajectory-bundle"
BUNDLE_FORMAT_VERSION = 1

#: PyUL save types handled here and their file stems (PyUL ``SaveFlags``/``SaveNames``).
SAVE_TYPE_STEMS = {"NBody": "NTM", "ULDCOM": "UCM", "DF": "DYF"}


def verify_length_conversion(atol_pc: float = 5e-5) -> float:
    """Check that one code length is 12133.0958 pc at 1e-21 eV; return the computed value."""
    value = pc_per_code_length(REFERENCE_AXION_MASS_EV)
    if abs(value - REFERENCE_PC_PER_CODE_LENGTH) > atol_pc:
        raise AssertionError(
            f"PyUL length unit at 1e-21 eV is {value!r} pc, expected {REFERENCE_PC_PER_CODE_LENGTH} pc."
        )
    return value


def resolve_axion_mass(
    reproducibility: Optional[dict], axion_mass_ev: Optional[float] = None
) -> Tuple[float, str]:
    """Return ``(axion_mass_ev, source)`` with the toolkit policy (:func:`.pyul_run.resolve_axion_mass`).

    The run's ``reproducibility.uldm`` takes precedence; ``axion_mass_ev`` is a fallback for runs
    that do not record the mass (if both exist they must agree, rel. 1e-6, or ``ValueError`` is
    raised); the ``PYUL_AXION_MASS_EV`` environment variable is used only as a last resort, with a
    ``RuntimeWarning``.
    """
    return _pr.resolve_axion_mass(reproducibility, axion_mass_ev, allow_env=True)


# ---------------------------------------------------------------------------
# Raw run directory reading
# ---------------------------------------------------------------------------
def find_run_dir(path: PathLike) -> Path:
    """Return the run directory (the folder holding ``config.uldm``) as a ``Path``.

    ``path`` may be the run directory itself, its ``Outputs`` folder or a PyUL ``save_path`` that
    contains exactly one run folder.  Unlike ``PyUL.Runs`` this never prompts; several candidate
    folders raise an error.  (Thin wrapper of :func:`.pyul_run.find_run_dir`.)
    """
    return Path(_pr.find_run_dir(path))


_FRAME_RE_TEMPLATE = r"^{stem}_#(\d+)\.{ext}$"


def _frame_files(run_dir: Path, save_type: str, save_format: str) -> List[Path]:
    stem = SAVE_TYPE_STEMS[save_type]
    pattern = re.compile(_FRAME_RE_TEMPLATE.format(stem=re.escape(stem), ext=re.escape(save_format)))
    for folder in (run_dir / "Outputs" / save_type, run_dir / "Outputs"):
        if not folder.is_dir():
            continue
        indexed = []
        for f in folder.iterdir():
            m = pattern.match(f.name)
            if m:
                indexed.append((int(m.group(1)), f))
        if indexed:
            indexed.sort()
            idx = [i for i, _ in indexed]
            if idx != list(range(len(idx))):
                missing = sorted(set(range(idx[-1] + 1)) - set(idx))
                raise ValueError(
                    f"{save_type} frames in {folder} are not contiguous from 0 (missing {missing[:10]}...)."
                )
            return [f for _, f in indexed]
    return []


def _load_frame(path: Path, save_format: str, save_type: str) -> np.ndarray:
    if save_format == "npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if save_format == "npz":
        with np.load(path, allow_pickle=False) as z:
            key = "arr_0" if "arr_0" in z.files else z.files[0]
            return np.asarray(z[key], dtype=np.float64)
    if save_format == "hdf5":
        try:
            import h5py  # noqa: WPS433 (optional dependency)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("h5py is required to read hdf5-format PyUL outputs") from exc
        with h5py.File(path, "r") as f:
            key = save_type if save_type in f else list(f.keys())[0]
            return np.asarray(f[key][()], dtype=np.float64)
    raise ValueError(f"Unsupported PyUL save format {save_format!r}")


def load_frames(run_dir: PathLike, save_type: str, save_format: Optional[str] = None) -> Optional[np.ndarray]:
    """Stack all frames of one save type into a ``(n_frames, n_values)`` float64 array.

    Returns ``None`` when the run has no frames of that type.  Each frame is flattened, so a
    two-body NBody state is ``(12,)`` per frame (x, y, z, vx, vy, vz of body 0, then body 1).
    """
    run_dir = Path(run_dir)
    if save_type not in SAVE_TYPE_STEMS:
        raise ValueError(f"save_type must be one of {sorted(SAVE_TYPE_STEMS)}")
    if save_format is None:
        save_format = _config_save_format(json.loads(_read_text(run_dir / "config.uldm")))
    files = _frame_files(run_dir, save_type, save_format)
    if not files:
        return None
    frames = [np.ravel(_load_frame(f, save_format, save_type)) for f in files]
    sizes = {fr.size for fr in frames}
    if len(sizes) != 1:
        raise ValueError(f"{save_type} frames in {run_dir} have inconsistent sizes {sorted(sizes)}")
    return np.vstack(frames)


def _config_save_format(config: dict) -> str:
    return _pr.config_save_format(config)


def _find_snapshot_times(run_dir: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    return _pr.find_snapshot_times(run_dir)


def _load_energies(run_dir: Path) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    outputs = run_dir / "Outputs"
    energies: Dict[str, np.ndarray] = {}
    uldm_mass = None
    if not outputs.is_dir():
        return energies, uldm_mass
    for f in sorted(outputs.glob("*list.npy")):
        energies[f.stem] = np.asarray(np.load(f, allow_pickle=False), dtype=np.float64)
    mass_file = outputs / "ULDMass.npy"
    if mass_file.is_file():
        uldm_mass = np.asarray(np.load(mass_file, allow_pickle=False), dtype=np.float64)
    return energies, uldm_mass


# ---------------------------------------------------------------------------
# Time axis (pyul_time_axis_myr is imported from pyul_run above)
# ---------------------------------------------------------------------------
_time_value_to_myr = _pr.time_value_to_myr


# ---------------------------------------------------------------------------
# Run container
# ---------------------------------------------------------------------------
@dataclass
class PyULRun:
    """Trajectory-level content of one PyUL run (raw directory or bundle).

    Arrays are in PyUL code units except ``times_myr``.  ``nbody`` has shape ``(T, 6 * n_bodies)``;
    ``uldcom`` ``(T, 3)``; ``df`` ``(T, k)``.  Use the accessor methods for physical units.
    """

    source: str
    config: dict
    reproducibility: dict
    config_text: str
    reproducibility_text: str
    axion_mass_ev: float
    axion_mass_source: str
    frame_index: np.ndarray
    nbody: Optional[np.ndarray]
    uldcom: Optional[np.ndarray] = None
    df: Optional[np.ndarray] = None
    energies: Dict[str, np.ndarray] = field(default_factory=dict)
    uldm_mass: Optional[np.ndarray] = None
    snapshot_times_myr: Optional[np.ndarray] = None
    snapshot_times_file: Optional[str] = None
    legacy_time_axis: bool = False
    times_myr: np.ndarray = field(default=None)  # type: ignore[assignment]
    time_axis_source: str = ""
    is_bundle: bool = False

    def __post_init__(self):
        self.frame_index = np.asarray(self.frame_index, dtype=np.int64)
        n = self.frame_index.size
        for name in ("nbody", "uldcom", "df"):
            arr = getattr(self, name)
            if arr is not None and arr.shape[0] != n:
                raise ValueError(f"{name} has {arr.shape[0]} frames but frame_index has {n}")
        self._set_time_axis()

    # -- time axis --------------------------------------------------------
    def _set_time_axis(self) -> None:
        n = self.frame_index.size
        n_needed = int(self.frame_index.max()) + 1 if n else 0

        def formula(legacy: bool) -> np.ndarray:
            return pyul_time_axis_myr(self.config, n_needed, legacy=legacy)[self.frame_index]

        if self.legacy_time_axis:
            self.times_myr = formula(True)
            self.time_axis_source = "legacy k*duration/(save_number+1) (old-notebook convention)"
            return
        if self.snapshot_times_myr is not None:
            snap = np.asarray(self.snapshot_times_myr, dtype=np.float64)
            if snap.size < n_needed:
                raise ValueError(
                    f"{self.snapshot_times_file} has {snap.size} entries but the run has {n_needed} frames."
                )
            snap = snap[self.frame_index]
            try:
                expected = formula(False)
            except ValueError:  # adaptive save count: nothing to compare against
                expected = None
            if expected is not None and n:
                dev = float(np.max(np.abs(snap - expected)))
                if dev > 1e-6 * max(1.0, float(np.max(np.abs(expected)))):
                    warnings.warn(
                        f"{self.snapshot_times_file} deviates from start + k*duration/save_number by up to "
                        f"{dev:.3g} Myr; using the file.",
                        RuntimeWarning,
                    )
            self.times_myr = snap
            self.time_axis_source = f"Outputs/{self.snapshot_times_file}"
            return
        self.times_myr = formula(False)
        self.time_axis_source = "start + k*duration/save_number (config.uldm)"

    def with_time_axis(self, legacy: bool) -> "PyULRun":
        """Return a copy that uses the requested time convention."""
        return replace(self, legacy_time_axis=bool(legacy), times_myr=None, time_axis_source="")

    def select_frames(self, mask_or_index) -> "PyULRun":
        """Return a copy restricted to a subset of frames (boolean mask or integer index)."""
        idx = np.arange(self.frame_index.size)[mask_or_index]

        def take(a):
            return None if a is None else a[idx]

        return replace(
            self,
            frame_index=self.frame_index[idx],
            nbody=take(self.nbody),
            uldcom=take(self.uldcom),
            df=take(self.df),
            energies={k: (v[self.frame_index[idx]] if v.shape[0] == self.n_saved_frames_total else v)
                      for k, v in self.energies.items()},
            uldm_mass=(self.uldm_mass[self.frame_index[idx]]
                       if self.uldm_mass is not None and self.uldm_mass.shape[0] == self.n_saved_frames_total
                       else self.uldm_mass),
            times_myr=None,
            time_axis_source="",
        )

    def window(self, t_min_myr: Optional[float] = None, t_max_myr: Optional[float] = None) -> "PyULRun":
        """Frames with ``t_min <= t <= t_max`` (on the current time axis)."""
        t = self.times_myr
        mask = np.ones(t.shape, dtype=bool)
        if t_min_myr is not None:
            mask &= t >= float(t_min_myr) - 1e-9
        if t_max_myr is not None:
            mask &= t <= float(t_max_myr) + 1e-9
        return self.select_frames(mask)

    # -- metadata ---------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return int(self.frame_index.size)

    @property
    def n_saved_frames_total(self) -> int:
        return int(self.config["Save Options"]["Number"]) + 1

    @property
    def n_bodies(self) -> int:
        if self.nbody is None:
            return 0
        if self.nbody.shape[1] % 6:
            raise ValueError(f"NBody state length {self.nbody.shape[1]} is not a multiple of 6")
        return self.nbody.shape[1] // 6

    @property
    def save_number(self) -> int:
        return int(self.config["Save Options"]["Number"])

    @property
    def duration_myr(self) -> float:
        d = self.config["Duration"]
        return _time_value_to_myr(d["Time Duration"], d.get("Time Units", ""))

    @property
    def resolution(self) -> int:
        return int(self.config["Spatial Resolution"])

    @property
    def box_length(self) -> Tuple[float, str]:
        b = self.config["Simulation Box"]
        return float(b["Box Length"]), str(b["Length Units"])

    @property
    def pyul_version(self):
        return self.reproducibility.get("Integrator Version", self.config.get("PyUL Version"))

    @property
    def pc_per_code(self) -> float:
        return pc_per_code_length(self.axion_mass_ev)

    def particle_masses_msun(self) -> np.ndarray:
        """Initial particle masses from ``config.uldm`` in solar masses."""
        block = self.config.get("Matter Particles", {})
        unit = block.get("Mass Units", "")
        out = []
        for cond in block.get("Condition", []):
            code = convert_to_code(float(cond[0]), unit, "m", self.axion_mass_ev)
            out.append(convert_from_code(code, "solar_masses", "m", self.axion_mass_ev))
        return np.asarray(out, dtype=np.float64)

    def summary(self) -> dict:
        return {
            "source": self.source,
            "is_bundle": self.is_bundle,
            "pyul_version": self.pyul_version,
            "resolution": self.resolution,
            "box_length": list(self.box_length),
            "duration_myr": self.duration_myr,
            "save_number": self.save_number,
            "n_frames": self.n_frames,
            "n_bodies": self.n_bodies,
            "particle_masses_msun": self.particle_masses_msun().tolist(),
            "axion_mass_ev": self.axion_mass_ev,
            "axion_mass_source": self.axion_mass_source,
            "pc_per_code_length": self.pc_per_code,
            "time_axis": self.time_axis_source,
            "t_first_myr": float(self.times_myr[0]) if self.n_frames else None,
            "t_last_myr": float(self.times_myr[-1]) if self.n_frames else None,
            "has_uldcom": self.uldcom is not None,
            "has_df": self.df is not None,
            "energies": sorted(self.energies),
        }

    # -- observables ------------------------------------------------------
    def _body_slice(self, body: int, offset: int) -> slice:
        if self.nbody is None:
            raise ValueError(f"{self.source} has no NBody output")
        if not 0 <= int(body) < self.n_bodies:
            raise IndexError(f"body {body} out of range (run has {self.n_bodies} bodies)")
        start = 6 * int(body) + offset
        return slice(start, start + 3)

    def positions_code(self, body: int = 0) -> np.ndarray:
        return self.nbody[:, self._body_slice(body, 0)]

    def velocities_code(self, body: int = 0) -> np.ndarray:
        return self.nbody[:, self._body_slice(body, 3)]

    def positions_pc(self, body: int = 0) -> np.ndarray:
        return self.positions_code(body) * self.pc_per_code

    def velocities_kms(self, body: int = 0) -> np.ndarray:
        return self.velocities_code(body) * kms_per_code_velocity(self.axion_mass_ev)

    def uldcom_pc(self) -> np.ndarray:
        if self.uldcom is None:
            raise ValueError(f"{self.source} has no ULDCOM output")
        return self.uldcom * self.pc_per_code

    def distance_from_origin_pc(self, body: int = 0) -> np.ndarray:
        """|x_BH| from the simulation origin (= the initial total centre of mass), in pc."""
        return np.linalg.norm(self.positions_code(body), axis=1) * self.pc_per_code

    def distance_from_uldm_com_pc(self, body: int = 0) -> np.ndarray:
        """|x_BH - x_ULDCOM| (BH relative to the ULDM centre of mass), in pc."""
        if self.uldcom is None:
            raise ValueError(f"{self.source} has no ULDCOM output")
        return np.linalg.norm(self.positions_code(body) - self.uldcom, axis=1) * self.pc_per_code

    def relative_position_pc(self, body: int = 0, frame: str = "uldcom") -> np.ndarray:
        """BH position in pc relative to ``'origin'`` or ``'uldcom'``."""
        if frame == "origin":
            return self.positions_pc(body)
        if frame == "uldcom":
            return (self.positions_code(body) - self.uldcom) * self.pc_per_code
        raise ValueError("frame must be 'origin' or 'uldcom'")


def distance_from_origin_pc(run: PyULRun, body: int = 0) -> np.ndarray:
    """Function form of :meth:`PyULRun.distance_from_origin_pc`."""
    return run.distance_from_origin_pc(body)


def distance_from_uldm_com_pc(run: PyULRun, body: int = 0) -> np.ndarray:
    """Function form of :meth:`PyULRun.distance_from_uldm_com_pc`."""
    return run.distance_from_uldm_com_pc(body)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_raw_run(path: PathLike, axion_mass_ev, legacy_time_axis: bool, load_energies: bool) -> PyULRun:
    meta = _pr.PyULRun.open(find_run_dir(path))
    run_dir = Path(meta.run_dir)
    mass, mass_source = resolve_axion_mass(meta.repro, axion_mass_ev)
    fmt = _config_save_format(meta.config)
    nbody = load_frames(run_dir, "NBody", fmt)
    uldcom = load_frames(run_dir, "ULDCOM", fmt)
    df = load_frames(run_dir, "DF", fmt)
    lengths = {name: a.shape[0] for name, a in (("NBody", nbody), ("ULDCOM", uldcom), ("DF", df)) if a is not None}
    if not lengths:
        raise FileNotFoundError(f"No NBody/ULDCOM/DF frames found under {run_dir / 'Outputs'}")
    n = min(lengths.values())
    if len(set(lengths.values())) != 1:
        warnings.warn(f"Frame counts differ between save types {lengths}; truncating to {n}.", RuntimeWarning)
        nbody = None if nbody is None else nbody[:n]
        uldcom = None if uldcom is None else uldcom[:n]
        df = None if df is None else df[:n]
    energies, uldm_mass = _load_energies(run_dir) if load_energies else ({}, None)
    snap, snap_file = _find_snapshot_times(run_dir)
    return PyULRun(
        source=str(run_dir),
        config=meta.config,
        reproducibility=meta.repro,
        config_text=meta.config_text,
        reproducibility_text=meta.repro_text,
        axion_mass_ev=mass,
        axion_mass_source=mass_source,
        frame_index=np.arange(n),
        nbody=nbody,
        uldcom=uldcom,
        df=df,
        energies=energies,
        uldm_mass=uldm_mass,
        snapshot_times_myr=snap,
        snapshot_times_file=snap_file,
        legacy_time_axis=legacy_time_axis,
    )


def load_run(
    path: PathLike,
    *,
    axion_mass_ev: Optional[float] = None,
    legacy_time_axis: bool = False,
    load_energies: bool = True,
) -> PyULRun:
    """Load a PyUL run directory, a ``save_path`` with one run folder, or a trajectory bundle (.npz).

    Parameters
    ----------
    axion_mass_ev
        Fallback axion mass for runs whose ``reproducibility.uldm`` lacks it.  If the run records a
        mass, this must agree with it.  ``PYUL_AXION_MASS_EV`` is used only if neither exists
        (last resort, with a ``RuntimeWarning``).
    legacy_time_axis
        Use the old notebooks' ``k * duration / (save_number + 1)`` time axis instead of PyUL's
        ``start + k * duration / save_number``.
    """
    p = Path(path)
    if p.is_file() and p.suffix == ".npz":
        return load_bundle(p, axion_mass_ev=axion_mass_ev, legacy_time_axis=legacy_time_axis)
    return _load_raw_run(p, axion_mass_ev, legacy_time_axis, load_energies)


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------
def _text_array(text: str) -> np.ndarray:
    return np.array(text, dtype=np.str_)


def export_bundle(
    run: Union[PyULRun, PathLike],
    out_path: PathLike,
    *,
    overwrite: bool = False,
    axion_mass_ev: Optional[float] = None,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Write a compressed single-file trajectory bundle for one run and return its path."""
    if not isinstance(run, PyULRun):
        run = load_run(run, axion_mass_ev=axion_mass_ev)
    if run.legacy_time_axis:
        run = run.with_time_axis(False)
    out = Path(out_path)
    if out.suffix != ".npz":
        raise ValueError("Bundle path must end in .npz")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_FORMAT_VERSION,
        "source_run_folder": Path(run.source).name,
        "axion_mass_ev": run.axion_mass_ev,
        "axion_mass_source": run.axion_mass_source,
        "save_format": _config_save_format(run.config),
        "n_frames": run.n_frames,
        "time_axis_of_times_myr": run.time_axis_source,
        "snapshot_times_file": run.snapshot_times_file,
        "units": "nbody/uldcom/df in PyUL code units; times_myr informational (recomputed on load)",
    }
    if extra_metadata:
        meta["extra"] = extra_metadata
    arrays = {
        "format": _text_array(BUNDLE_FORMAT),
        "format_version": np.array(BUNDLE_FORMAT_VERSION, dtype=np.int64),
        "metadata_json": _text_array(json.dumps(meta, sort_keys=True)),
        "config_json": _text_array(run.config_text),
        "reproducibility_json": _text_array(run.reproducibility_text),
        "frame_index": run.frame_index.astype(np.int64),
        "times_myr": np.asarray(run.times_myr, dtype=np.float64),
        "particle_masses_msun": run.particle_masses_msun(),
    }
    for name in ("nbody", "uldcom", "df", "uldm_mass", "snapshot_times_myr"):
        value = getattr(run, name)
        if value is not None:
            arrays[name] = np.asarray(value, dtype=np.float64)
    for key, value in run.energies.items():
        arrays["energy__" + key] = np.asarray(value, dtype=np.float64)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out)
    return out


def load_bundle(
    path: PathLike, *, axion_mass_ev: Optional[float] = None, legacy_time_axis: bool = False
) -> PyULRun:
    """Load a bundle written by :func:`export_bundle`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        if "format" not in z.files or str(z["format"]) != BUNDLE_FORMAT:
            raise ValueError(f"{path} is not a {BUNDLE_FORMAT} file")
        version = int(z["format_version"])
        if version > BUNDLE_FORMAT_VERSION:
            raise ValueError(f"{path} has bundle format version {version}; this reader supports <= {BUNDLE_FORMAT_VERSION}")
        meta = json.loads(str(z["metadata_json"]))
        config_text = str(z["config_json"])
        repro_text = str(z["reproducibility_json"])
        get = lambda k: (np.array(z[k]) if k in z.files else None)  # noqa: E731
        energies = {k[len("energy__"):]: np.array(z[k]) for k in z.files if k.startswith("energy__")}
        frame_index = np.array(z["frame_index"])
        nbody, uldcom, df = get("nbody"), get("uldcom"), get("df")
        uldm_mass, snap = get("uldm_mass"), get("snapshot_times_myr")
        stored_times = get("times_myr")
    config = json.loads(config_text)
    repro = json.loads(repro_text) if repro_text else {}
    if axion_mass_from_reproducibility(repro) is None and axion_mass_ev is None:
        # bundle metadata keeps the mass that was used at export time (explicit argument then)
        axion_mass_ev = meta.get("axion_mass_ev")
        mass, source = resolve_axion_mass(repro, axion_mass_ev)
        source = "bundle metadata (" + str(meta.get("axion_mass_source")) + ")"
    else:
        mass, source = resolve_axion_mass(repro, axion_mass_ev)
    run = PyULRun(
        source=str(path),
        config=config,
        reproducibility=repro,
        config_text=config_text,
        reproducibility_text=repro_text,
        axion_mass_ev=mass,
        axion_mass_source=source,
        frame_index=frame_index,
        nbody=nbody,
        uldcom=uldcom,
        df=df,
        energies=energies,
        uldm_mass=uldm_mass,
        snapshot_times_myr=snap,
        snapshot_times_file=meta.get("snapshot_times_file"),
        legacy_time_axis=legacy_time_axis,
        is_bundle=True,
    )
    if stored_times is not None and not legacy_time_axis:
        if stored_times.shape != run.times_myr.shape or not np.allclose(stored_times, run.times_myr, rtol=0, atol=1e-9):
            raise ValueError(f"{path}: stored times_myr disagree with the times recomputed from config.uldm")
    return run


# ---------------------------------------------------------------------------
# Command line: python -m Eig_ULDM_packages.trajectories {info,export} ...
# ---------------------------------------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PyUL trajectory loader / bundle exporter")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_info = sub.add_parser("info", help="print a JSON summary of a run directory or bundle")
    p_info.add_argument("path")
    p_info.add_argument("--axion-mass-ev", type=float, default=None)
    p_info.add_argument("--legacy-time-axis", action="store_true")
    p_exp = sub.add_parser("export", help="write a single-file trajectory bundle (.npz)")
    p_exp.add_argument("run_dir")
    p_exp.add_argument("out")
    p_exp.add_argument("--axion-mass-ev", type=float, default=None)
    p_exp.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "info":
        run = load_run(args.path, axion_mass_ev=args.axion_mass_ev, legacy_time_axis=args.legacy_time_axis)
        print(json.dumps(run.summary(), indent=2))
    else:
        out = export_bundle(args.run_dir, args.out, overwrite=args.overwrite, axion_mass_ev=args.axion_mass_ev)
        print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
