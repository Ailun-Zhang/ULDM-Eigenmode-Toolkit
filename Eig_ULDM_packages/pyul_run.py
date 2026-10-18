"""Read a PyUL run directory: the single run loader of the toolkit.

Every toolkit module that reads a PyUL run (the c_nlm pipeline
:mod:`Eig_ULDM_packages.pipeline`, the in-simulation recorder
:mod:`Eig_ULDM_packages.cnlm_inline`, the trajectory loader
:mod:`Eig_ULDM_packages.trajectories`, ``compute_coeff.py`` and the paper
figure scripts) uses the functions of this module for

* locating the run directory (:func:`find_run_dir`),
* reading ``config.uldm`` / ``reproducibility.uldm`` (:meth:`PyULRun.open`),
* PyUL's unit system (:func:`code_units` and friends; constants copied from
  ``PyUltraLight2/Current_SK.py``),
* the axion mass (:func:`resolve_axion_mass`, policy below),
* the snapshot time axis (:func:`pyul_time_axis_myr`, :func:`find_snapshot_times`,
  :meth:`PyULRun.snapshot_times_myr`) and
* the resampled-box geometry (:func:`rs_geometry`).

It needs only NumPy and the standard library (no PyUL, no h5py, no
pyshtools), and it never imports :mod:`Eig_ULDM_packages.units` at import
time.

A PyUL run directory looks like::

    <save_path>/<YYYYMMDD_HHMMSS>@<resol>/
        config.uldm            (JSON: duration, save number, box, solitons, ...)
        reproducibility.uldm   (JSON: integrator version, axion mass, flags, ...)
        Outputs/3WfnRS/P3R_#000.npy ...

Axion mass policy
-----------------
1. The mass recorded in the run's ``reproducibility.uldm`` (``"Axion Mass"``
   in kg and/or ``"m22"``; both must agree to 1e-6 when both exist).
2. Otherwise an explicit argument (``axion_mass_ev=`` / ``--axion-mass-ev``).
   An explicit value that disagrees with the record raises ``ValueError``.
3. Otherwise, as a last resort, the environment variable
   ``PYUL_AXION_MASS_EV`` with a ``RuntimeWarning`` (it is a process-wide
   setting that silently changes every unit; prefer the record or the
   argument).
4. Otherwise ``ValueError``.

Resampled-box (3WfnRS) geometry
-------------------------------
PyUL 30.40 and earlier do **not** record ``Length_Ratio``, ``resolR`` or
``CellCenteredGrid``, so the side of the resampled cube cannot be recovered
from the metadata and must be given explicitly.  PyUL_SK 30.50 writes them to
``reproducibility.uldm`` under ``"SK Grid and Output Settings" ->
"Resampled Box"`` (keys ``"resolR"``, ``"Length_Ratio"``, ``"Legacy Resample
Box"``, ``"Side (code units)"``, ``"Side (kpc)"``, ``"Spacing (code units)"``,
``"Spacing (kpc)"``) and ``"Cell Centered Grid"``; :func:`rs_geometry` reads
them when present.  Note that with ``CellCenteredGrid=True`` PyUL 30.38-30.40
resampled onto a cube of side ``(L - dx)*Length_Ratio``, not
``L*Length_Ratio``.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "AXION_MASS_ENV",
    "EV_KG",
    "TIME_UNITS_TO_MYR",
    "PyULRun",
    "RSGeometry",
    "find_run_dir",
    "read_run_metadata",
    "rs_geometry",
    "code_units",
    "pc_per_code_length",
    "myr_per_code_time",
    "msun_per_code_mass",
    "kms_per_code_velocity",
    "convert_to_code",
    "convert_from_code",
    "axion_mass_from_reproducibility",
    "resolve_axion_mass",
    "config_save_format",
    "time_value_to_myr",
    "pyul_time_axis_myr",
    "find_snapshot_times",
]

PathLike = Union[str, Path]

#: Environment variable consulted only as the last resort of the axion-mass policy.
AXION_MASS_ENV = "PYUL_AXION_MASS_EV"

TIME_UNITS_TO_MYR = {"Myr": 1.0, "Gyr": 1.0e3, "kyr": 1.0e-3, "yr": 1.0e-6}

# ---------------------------------------------------------------------------
# PyUL unit system (constants copied verbatim from PyUltraLight2/Current_SK.py)
# ---------------------------------------------------------------------------
HBAR = 1.0545718e-34  # J s
PARSEC = 3.0857e16  # m
LIGHT_YEAR = 9.4607e15  # m
SOLAR_MASS = 1.989e30  # kg
G_NEWTON = 6.67e-11  # m^3 kg^-1 s^-2
OMEGA_M0 = 0.31
H_0 = 67.7 / (PARSEC * 1e3)  # s^-1
EV_KG = 1.78266191e-36  # kg per eV/c^2 (PyUL `eV`, toolkit `units.eV`)
SECONDS_PER_YEAR = 60 * 60 * 24 * 365  # PyUL uses a 365-day year

REFERENCE_AXION_MASS_EV = 1e-21

_TIME_SECONDS = {
    "s": 1.0,
    "SI": 1.0,
    "yr": SECONDS_PER_YEAR,
    "kyr": SECONDS_PER_YEAR * 1e3,
    "Myr": SECONDS_PER_YEAR * 1e6,
    "Gyr": SECONDS_PER_YEAR * 1e9,
}
_LENGTH_METRES = {
    "m": 1.0,
    "SI": 1.0,
    "km": 1e3,
    "pc": PARSEC,
    "kpc": 1e3 * PARSEC,
    "Mpc": 1e6 * PARSEC,
    "ly": LIGHT_YEAR,
}
_MASS_KG = {"kg": 1.0, "SI": 1.0, "solar_masses": SOLAR_MASS, "M_solar_masses": 1e6 * SOLAR_MASS}
_SPEED_MS = {"m/s": 1.0, "SI": 1.0, "km/s": 1e3, "km/h": 1e3 / 3600.0}


def _check_positive(value, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive number, got {value!r}") from None
    if not (out > 0 and math.isfinite(out)):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return out


def code_units(axion_mass_ev: float) -> Dict[str, float]:
    """Return PyUL's code units (SI) for a given axion mass in eV.

    Returns a dict with ``length_m``, ``time_s``, ``mass_kg`` and ``velocity_ms``.
    The time unit does not depend on the axion mass.
    """
    m = _check_positive(axion_mass_ev, "axion_mass_ev") * EV_KG
    time_s = (3 * H_0 ** 2 * OMEGA_M0 / (8 * math.pi)) ** -0.5
    length_m = (8 * math.pi * HBAR ** 2 / (3 * m ** 2 * H_0 ** 2 * OMEGA_M0)) ** 0.25
    mass_kg = (3 * H_0 ** 2 * OMEGA_M0 / (8 * math.pi)) ** 0.25 * HBAR ** 1.5 / (m ** 1.5 * G_NEWTON)
    return {"length_m": length_m, "time_s": time_s, "mass_kg": mass_kg, "velocity_ms": length_m / time_s}


def pc_per_code_length(axion_mass_ev: float) -> float:
    """Parsec per PyUL code length."""
    return code_units(axion_mass_ev)["length_m"] / PARSEC


def myr_per_code_time() -> float:
    """Myr per PyUL code time (independent of the axion mass)."""
    return code_units(REFERENCE_AXION_MASS_EV)["time_s"] / _TIME_SECONDS["Myr"]


def msun_per_code_mass(axion_mass_ev: float) -> float:
    """Solar masses per PyUL code mass."""
    return code_units(axion_mass_ev)["mass_kg"] / SOLAR_MASS


def kms_per_code_velocity(axion_mass_ev: float) -> float:
    """km/s per PyUL code velocity."""
    return code_units(axion_mass_ev)["velocity_ms"] / 1e3


def _unit_table(kind: str) -> Dict[str, float]:
    tables = {"l": _LENGTH_METRES, "t": _TIME_SECONDS, "m": _MASS_KG, "v": _SPEED_MS}
    if kind not in tables:
        raise ValueError(f"Unsupported quantity kind {kind!r}; use one of 'l', 't', 'm', 'v'.")
    return tables[kind]


def _code_si(kind: str, axion_mass_ev: Optional[float]) -> float:
    if kind == "t":
        return code_units(REFERENCE_AXION_MASS_EV)["time_s"]
    if axion_mass_ev is None:
        raise ValueError(f"Converting a '{kind}' quantity needs the axion mass (axion_mass_ev).")
    units = code_units(axion_mass_ev)
    return {"l": units["length_m"], "m": units["mass_kg"], "v": units["velocity_ms"]}[kind]


def convert_to_code(value, unit: str, kind: str, axion_mass_ev: Optional[float] = None):
    """Convert ``value`` in ``unit`` to PyUL code units (mirrors PyUL ``convert``; '' = code units)."""
    if unit == "":
        return value
    table = _unit_table(kind)
    if unit not in table:
        raise ValueError(f"Unsupported unit {unit!r} for kind {kind!r}.")
    return value * table[unit] / _code_si(kind, axion_mass_ev)


def convert_from_code(value, unit: str, kind: str, axion_mass_ev: Optional[float] = None):
    """Convert ``value`` in PyUL code units to ``unit`` (mirrors PyUL ``convert_back``)."""
    if unit == "":
        return value
    table = _unit_table(kind)
    if unit not in table:
        raise ValueError(f"Unsupported unit {unit!r} for kind {kind!r}.")
    return value * _code_si(kind, axion_mass_ev) / table[unit]


# ---------------------------------------------------------------------------
# Axion mass (single policy for the whole toolkit)
# ---------------------------------------------------------------------------
def _nice(x: float) -> float:
    """Remove float noise from a recorded value (e.g. 5.000000000000004 -> 5.0, 1.0000000000000001e-21 -> 1e-21).

    The value is snapped to its 12-significant-digit form only if that changes it by less than
    1e-13 relative, so genuine digits (e.g. 4.973958333333333 = 5*191/192) are kept.
    """
    x = float(x)
    r = float(f"{x:.12g}")
    return r if abs(r - x) <= 1e-13 * abs(x) else x


def axion_mass_from_reproducibility(reproducibility: Optional[dict]) -> Optional[float]:
    """Axion mass in eV recorded in a parsed ``reproducibility.uldm`` (or ``None``).

    PyUL records ``"Axion Mass"`` in kg and ``"m22"`` (mass / 1e-22 eV).  When both are present
    they must agree to 1e-6 (``ValueError`` otherwise).  Float noise of the kg / m22 round trip is
    removed (1.7826619099999998e-57 kg -> 1e-21 eV; ``10.0 * 1e-22`` -> 1e-21).
    """
    if not reproducibility:
        return None
    candidates = []
    kg = reproducibility.get("Axion Mass")
    if kg is not None:
        candidates.append(("Axion Mass", _nice(float(kg) / EV_KG)))
    m22 = reproducibility.get("m22")
    if m22 is not None:
        candidates.append(("m22", _nice(float(m22) * 1e-22)))
    if not candidates:
        return None
    values = [v for _, v in candidates]
    if len(values) == 2 and not math.isclose(values[0], values[1], rel_tol=1e-6):
        raise ValueError(f"Inconsistent axion mass in reproducibility.uldm: {candidates}")
    return values[0]


def resolve_axion_mass(
    reproducibility: Optional[dict],
    axion_mass_ev: Optional[float] = None,
    *,
    allow_env: bool = True,
    env_var: str = AXION_MASS_ENV,
) -> Tuple[float, str]:
    """Return ``(axion_mass_ev, source)`` following the policy of the module docstring.

    ``source`` is ``'reproducibility.uldm'``, ``'argument'`` or ``'environment variable <name>'``.
    ``allow_env=False`` skips step 3 (the environment variable).
    """
    recorded = axion_mass_from_reproducibility(reproducibility)
    if recorded is not None:
        if axion_mass_ev is not None and not math.isclose(float(axion_mass_ev), recorded, rel_tol=1e-6):
            raise ValueError(
                f"axion_mass_ev={axion_mass_ev} disagrees with the run's reproducibility.uldm ({recorded} eV)."
            )
        env = os.environ.get(env_var, "").strip()
        if env:
            try:
                env_val = float(env)
            except ValueError:
                env_val = None
            if env_val is None or not math.isclose(env_val, recorded, rel_tol=1e-6):
                warnings.warn(f"{env_var}={env} differs from the run's recorded axion mass {recorded:g} eV; "
                              "using the recorded value.", RuntimeWarning, stacklevel=2)
        return recorded, "reproducibility.uldm"
    if axion_mass_ev is not None:
        return _check_positive(axion_mass_ev, "axion_mass_ev"), "argument"
    if allow_env:
        env = os.environ.get(env_var, "").strip()
        if env:
            value = _check_positive(env, env_var)
            warnings.warn(
                f"The run does not record its axion mass; using {env_var}={env} eV from the environment "
                "(last resort). Pass the axion mass explicitly to silence this warning.",
                RuntimeWarning, stacklevel=2)
            return value, f"environment variable {env_var}"
    raise ValueError(
        "The run does not record its axion mass (no 'Axion Mass'/'m22' in reproducibility.uldm); "
        "pass axion_mass_ev explicitly (the paper runs used 1e-21 eV)"
        + (f" or set {env_var}." if allow_env else ".")
    )


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------
def find_run_dir(path: PathLike) -> str:
    """Return the PyUL run directory (the folder holding ``config.uldm``) as a string.

    ``path`` may be the run directory, its ``Outputs`` folder, or a PyUL ``save_path`` that
    contains exactly one run folder.  Unlike ``PyUL.Runs`` this never prompts; several candidate
    folders raise ``ValueError``, no candidate ``FileNotFoundError``.
    """
    p = Path(os.path.abspath(os.fspath(path)))
    if (p / "config.uldm").is_file():
        return str(p)
    if p.name == "Outputs" and (p.parent / "config.uldm").is_file():
        return str(p.parent)
    if not p.is_dir():
        raise FileNotFoundError(f"Run path does not exist: {p}")
    candidates = sorted(d for d in p.iterdir() if d.is_dir() and (d / "config.uldm").is_file())
    if len(candidates) == 1:
        return str(candidates[0])
    if not candidates:
        raise FileNotFoundError(
            f"{p} is not a PyUL run directory (no config.uldm here, in its parent or in its subdirectories).")
    names = ", ".join(d.name for d in candidates)
    raise ValueError(f"{p} contains several run folders ({names}); pass the run folder explicitly.")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def read_run_metadata(run_dir: PathLike) -> Tuple[Dict[str, Any], str, Dict[str, Any], str]:
    """``(config, config_text, reproducibility, reproducibility_text)`` of a run directory.

    ``config.uldm`` must exist; a missing ``reproducibility.uldm`` gives ``({}, '')``.
    """
    run_dir = os.fspath(run_dir)
    config_text = _read_text(os.path.join(run_dir, "config.uldm"))
    config = json.loads(config_text)
    repro_path = os.path.join(run_dir, "reproducibility.uldm")
    repro_text = _read_text(repro_path) if os.path.isfile(repro_path) else ""
    repro = json.loads(repro_text) if repro_text.strip() else {}
    return config, config_text, repro, repro_text


def config_save_format(config: dict) -> str:
    """PyUL save format recorded in ``config.uldm`` (default ``'npy'``)."""
    return str(config.get("Save Options", {}).get("Format", "npy"))


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------
def time_value_to_myr(value: float, unit: str) -> float:
    """A PyUL duration in ``unit`` ('' = code units) in Myr (365-day years, as PyUL)."""
    if unit == "":
        return float(value) * myr_per_code_time()
    if unit not in _TIME_SECONDS:
        raise ValueError(f"Unsupported time unit {unit!r} in config.uldm")
    return float(value) * _TIME_SECONDS[unit] / _TIME_SECONDS["Myr"]


def pyul_time_axis_myr(config: dict, n_frames: int, legacy: bool = False) -> np.ndarray:
    """Snapshot times in Myr from ``config.uldm``.

    ``legacy=False`` (default): ``t_k = start + k * duration / save_number`` (what PyUL does).
    ``legacy=True``: ``t_k = k * duration / (save_number + 1)``, the convention of the old analysis
    notebooks (start time ignored, as they did).  Use only to reproduce figures made with it.
    """
    duration_block = config["Duration"]
    units = duration_block.get("Time Units", "")
    duration_myr = time_value_to_myr(duration_block["Time Duration"], units)
    start_myr = time_value_to_myr(duration_block.get("Start Time", 0.0), units)
    save_number = int(config["Save Options"]["Number"])
    if save_number <= 0:
        raise ValueError(
            "config.uldm has Save Options Number <= 0 (adaptive save count); the snapshot times "
            "cannot be reconstructed from the config. Use a run with Outputs/snapshot_times_*.npy."
        )
    k = np.arange(int(n_frames), dtype=np.float64)
    if legacy:
        return k * duration_myr / (save_number + 1)
    return start_myr + k * duration_myr / save_number


_SNAPSHOT_RE = re.compile(r"^snapshot_times_(?P<unit>[A-Za-z]+)\.npy$")


def find_snapshot_times(run_dir: PathLike) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Explicit snapshot times ``Outputs/snapshot_times_<unit>.npy`` in Myr, and the file name.

    PyUL_SK >= 30.50 writes ``snapshot_times_Myr.npy`` and ``snapshot_times_code.npy`` (the
    times of the snapshots actually written, including the start time).  ``Myr`` is preferred.
    Returns ``(None, None)`` when there is no such file.
    """
    outputs = os.path.join(os.fspath(run_dir), "Outputs")
    if not os.path.isdir(outputs):
        return None, None
    found = {}
    for name in os.listdir(outputs):
        m = _SNAPSHOT_RE.match(name)
        if m:
            found[m.group("unit")] = os.path.join(outputs, name)
    if not found:
        return None, None
    unit = "Myr" if "Myr" in found else sorted(found)[0]
    raw = np.asarray(np.load(found[unit], allow_pickle=False), dtype=np.float64).ravel()
    if unit == "Myr":
        myr = raw
    elif unit == "code":
        myr = raw * myr_per_code_time()
    elif unit in _TIME_SECONDS:
        myr = raw * _TIME_SECONDS[unit] / _TIME_SECONDS["Myr"]
    else:
        raise ValueError(f"Unknown unit in snapshot-time file name {os.path.basename(found[unit])}")
    return myr, os.path.basename(found[unit])


# ---------------------------------------------------------------------------
# Resampled-box geometry
# ---------------------------------------------------------------------------
@dataclass
class RSGeometry:
    """Geometry of a resampled (3WfnRS) cube."""

    side: float
    resol: int
    units: str
    source: str
    dx: float = 0.0
    legacy_resample_box: Optional[bool] = None
    cell_centered: Optional[bool] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.dx:
            self.dx = self.side / self.resol

    @property
    def half_side(self) -> float:
        return self.side / 2.0


# ---------------------------------------------------------------------------
# Run metadata container
# ---------------------------------------------------------------------------
@dataclass
class PyULRun:
    """Parsed PyUL run directory (metadata only; see :mod:`.trajectories` for the frames)."""

    run_dir: str
    config: Dict[str, Any]
    repro: Dict[str, Any]
    config_text: str = ""
    repro_text: str = ""

    @classmethod
    def open(cls, path: PathLike) -> "PyULRun":
        run_dir = find_run_dir(path)
        config, config_text, repro, repro_text = read_run_metadata(run_dir)
        return cls(run_dir=run_dir, config=config, repro=repro, config_text=config_text,
                   repro_text=repro_text)

    # ------------------------------------------------------------------
    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.run_dir, "Outputs")

    @property
    def integrator_version(self) -> Optional[float]:
        v = self.repro.get("Integrator Version", self.config.get("PyUL Version"))
        return None if v is None else float(v)

    @property
    def save_number(self) -> int:
        """``Save Options Number`` as recorded in config.uldm (``-1`` = PyUL saved every step)."""
        return int(self.config["Save Options"]["Number"])

    def effective_save_number(self, prefix: str = "3WfnRS/P3R_#", extra_indices: Iterable[int] = ()) -> int:
        """Highest save index of the run.

        Equal to :attr:`save_number` when config.uldm records one (>= 0).  For ``Number = -1`` PyUL
        replaces the value in memory by the number of steps, which config.uldm does not keep; it is
        then taken from the saves that exist: ``Outputs/snapshot_times_Myr.npy`` (PyUL_SK >= 30.50),
        the snapshot files ``Outputs/<prefix>NNN.npy`` and ``extra_indices`` (e.g. inline
        checkpoints), whichever reaches furthest.  Raises ``ValueError`` if none exists.
        """
        n = self.save_number
        if n >= 0:
            return n
        cands = [int(k) for k in extra_indices]
        cands += self.available_snapshots(prefix)
        snap, _ = find_snapshot_times(self.run_dir)
        if snap is not None and snap.size:
            cands.append(int(snap.size) - 1)
        if not cands:
            raise ValueError(f"config.uldm has Save Options Number = {n} (save every step) and the run has no "
                             "Outputs/snapshot_times_Myr.npy, snapshot files or checkpoints to count the saves")
        return max(cands)

    @property
    def save_format(self) -> str:
        return config_save_format(self.config)

    @property
    def duration(self) -> float:
        return float(self.config["Duration"]["Time Duration"])

    @property
    def time_units(self) -> str:
        return str(self.config["Duration"].get("Time Units", "Myr"))

    @property
    def start_time(self) -> float:
        return float(self.config["Duration"].get("Start Time", 0.0))

    @property
    def duration_myr(self) -> float:
        u = self.time_units
        if u not in TIME_UNITS_TO_MYR:
            raise ValueError(f"unsupported PyUL time unit {u!r}; supported: {sorted(TIME_UNITS_TO_MYR)}")
        return self.duration * TIME_UNITS_TO_MYR[u]

    @property
    def box_length(self) -> float:
        return float(self.config["Simulation Box"]["Box Length"])

    @property
    def length_units(self) -> str:
        return str(self.config["Simulation Box"].get("Length Units", "kpc"))

    @property
    def resol(self) -> int:
        return int(self.config["Spatial Resolution"])

    def axion_mass_ev(self) -> Optional[float]:
        """Axion mass in eV recorded in reproducibility.uldm (``None`` if not recorded)."""
        return axion_mass_from_reproducibility(self.repro)

    def resolve_axion_mass(self, axion_mass_ev: Optional[float] = None, *,
                           allow_env: bool = True) -> Tuple[float, str]:
        """``(axion_mass_ev, source)`` with the toolkit policy (see :func:`resolve_axion_mass`)."""
        return resolve_axion_mass(self.repro, axion_mass_ev, allow_env=allow_env)

    def soliton_masses(self) -> List[float]:
        sol = self.config.get("ULDM Solitons", {})
        return [float(c[0]) for c in sol.get("Condition", [])]

    @property
    def soliton_mass_units(self) -> str:
        return str(self.config.get("ULDM Solitons", {}).get("Mass Units", "M_solar_masses"))

    @property
    def embedded_solitons(self) -> list:
        return list(self.config.get("ULDM Solitons", {}).get("Embedded", []) or [])

    def snapshot_path(self, prefix: str, index: int, ext: str = ".npy") -> str:
        """``Outputs/<prefix><NNN>.npy`` with PyUL's ``{index:03d}`` numbering."""
        return os.path.join(self.outputs_dir, f"{prefix}{int(index):03d}{ext}")

    def available_snapshots(self, prefix: str = "3WfnRS/P3R_#", ext: str = ".npy") -> List[int]:
        pat = os.path.join(self.outputs_dir, prefix + "*" + ext)
        rx = re.compile(re.escape(os.path.basename(prefix)) + r"(\d+)" + re.escape(ext) + "$")
        out = []
        for p in glob.glob(pat):
            mt = rx.search(os.path.basename(p))
            if mt:
                out.append(int(mt.group(1)))
        return sorted(out)

    def snapshot_times_myr(self, indices: Sequence[int], return_source: bool = False):
        """Times of PyUL saves ``k`` in Myr.

        PyUL_SK >= 30.50 runs: the real save times written by PyUL,
        ``Outputs/snapshot_times_Myr.npy`` (``start + k*duration/save_number``, computed
        by PyUL in code units and converted to Myr), when the file covers ``indices``.
        Older runs: ``linspace(0, duration, save_number+1)[k]`` -- the convention of
        ``load_a_lm(final_time=duration)`` for the full range 0..save_number.
        ``return_source=True`` returns ``(times, description)``.
        """
        idx = np.asarray(list(indices), dtype=np.int64)
        if np.any(idx < 0) or (self.save_number >= 0 and np.any(idx > self.save_number)):
            raise ValueError(f"snapshot indices must lie in 0..{self.save_number}")
        snap, name = find_snapshot_times(self.run_dir)
        if snap is not None and (idx.size == 0 or int(idx.max()) < snap.size):
            out, src = np.asarray(snap[idx], dtype=np.float64), f"Outputs/{name} (PyUL save times)"
        elif self.save_number < 0:
            # Number = -1: the save interval is the step size, which config.uldm does not record
            raise ValueError(f"config.uldm has Save Options Number = {self.save_number} (save every step); the "
                             "save times can only be taken from Outputs/snapshot_times_Myr.npy (PyUL_SK >= 30.50), "
                             "which is missing or does not cover the requested snapshots")
        else:
            out = np.linspace(0.0, self.duration_myr, self.save_number + 1, endpoint=True)[idx]
            src = "linspace(0, duration, save_number+1)[snapshot_indices]"
        return (out, src) if return_source else out

    def rs_record(self) -> Optional[Dict[str, Any]]:
        """The 'Resampled Box' block written by PyUL_SK >= 30.50, if present."""
        sk = self.repro.get("SK Grid and Output Settings")
        if isinstance(sk, dict) and isinstance(sk.get("Resampled Box"), dict):
            rec = dict(sk["Resampled Box"])
            if "Cell Centered Grid" in sk:
                rec["Cell Centered Grid"] = sk["Cell Centered Grid"]
            return rec
        return None


def rs_geometry(
    run: Optional[PyULRun] = None,
    *,
    rs_side: Optional[float] = None,
    rs_resol: Optional[int] = None,
    rs_dx: Optional[float] = None,
    units: Optional[str] = None,
    cube_resol: Optional[int] = None,
) -> RSGeometry:
    """Resolve the resampled-cube geometry.

    Priority: explicit ``rs_dx`` (with ``rs_resol`` or ``cube_resol``) or
    explicit ``rs_side`` > the PyUL >= 30.50 record in reproducibility.uldm.
    Raises ``ValueError`` if neither is available (PyUL <= 30.40 runs), rather
    than guessing.  ``units`` defaults to the run's box length units ('kpc').
    """
    units = units or (run.length_units if run is not None else "kpc")
    resol = rs_resol or cube_resol
    if rs_dx is not None:
        if resol is None:
            raise ValueError("rs_dx needs the cube resolution (rs_resol)")
        return RSGeometry(side=float(rs_dx) * int(resol), resol=int(resol), units=units,
                          source="explicit dx", dx=float(rs_dx))
    if rs_side is not None:
        if resol is None:
            raise ValueError("rs_side needs the cube resolution (rs_resol)")
        return RSGeometry(side=float(rs_side), resol=int(resol), units=units, source="explicit side")
    rec = run.rs_record() if run is not None else None
    if rec is not None:
        r_resol = int(rec["resolR"])
        if cube_resol is not None and int(cube_resol) != r_resol:
            raise ValueError(f"reproducibility.uldm says resolR={r_resol} but the cubes are {cube_resol}^3")
        if units == "kpc" and "Side (kpc)" in rec:
            side = _nice(rec["Side (kpc)"])
        elif "Side (code units)" in rec:
            from . import units as _units
            side = _nice(_units.convert_back(float(rec["Side (code units)"]), units, 'l'))
        else:
            raise ValueError("the 'Resampled Box' record has no side length")
        return RSGeometry(side=side, resol=r_resol, units=units, source="reproducibility.uldm",
                          legacy_resample_box=rec.get("Legacy Resample Box"),
                          cell_centered=rec.get("Cell Centered Grid"), raw=rec)
    raise ValueError(
        "The resampled (3WfnRS) box geometry is not recorded in this run's reproducibility.uldm "
        "(PyUL <= 30.40). Give it explicitly: --rs-side <side> (and --rs-resol) or --rs-dx <spacing>, "
        "in the run's length units. Note: with CellCenteredGrid=True, PyUL 30.38-30.40 used a side of "
        "(L - L/N)*Length_Ratio."
    )
