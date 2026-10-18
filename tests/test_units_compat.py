"""T1/T3/T13: compat helpers and units."""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO
from Eig_ULDM_packages import _compat


def _run(code, env_extra=None, drop_env=()):
    env = dict(os.environ)
    for k in drop_env:
        env.pop(k, None)
    env.update(env_extra or {})
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env)


def test_trapezoid_matches_formula():
    rng = np.random.default_rng(0)
    y = rng.normal(size=101) + 1j * rng.normal(size=101)
    dx = 0.37
    expected = np.add.reduce(dx * (y[1:] + y[:-1]) / 2.0)
    assert _compat.trapezoid(y, dx=dx) == expected


def test_mp_context_and_workers(monkeypatch):
    ctx = _compat.mp_context()
    assert ctx.get_start_method() in ("fork", "spawn")
    if sys.platform == "win32":
        assert ctx.get_start_method() == "spawn"
    with pytest.raises(ValueError):
        _compat.mp_context("no-such-method")
    monkeypatch.setenv("ULDM_NUM_WORKERS", "3")
    assert _compat.default_workers() == 3
    assert _compat.default_workers(5) == 5
    monkeypatch.delenv("ULDM_NUM_WORKERS")
    # Higher-priority scheduler variables (set inside PBS/Slurm jobs) must not leak into this check.
    for name in ("PBS_NCPUS", "NCPUS", "OMP_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert _compat.default_workers() == 1


def test_convert_back_c_roundtrip():
    from Eig_ULDM_packages import units
    for v in (1e-3, 0.5):
        assert abs(units.convert_back(units.convert(v, "c", "v"), "c", "v") - v) < 1e-12 * v
    assert abs(units.convert_between(1.0, "c", "km/s", "v") - 299792.458) < 1e-6


def test_default_mass_warning_once():
    code = ("import warnings\nwarnings.simplefilter('always')\n"
            "import Eig_ULDM_packages.units as u\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    u.convert(1.0, 'kpc', 'l'); u.convert(2.0, 'kpc', 'l'); u.convert_back(1.0, 'kpc', 'l')\n"
            "print(sum('PYUL_AXION_MASS_EV' in str(x.message) for x in w), u.axion_mass_source())\n")
    r = _run(code, drop_env=("PYUL_AXION_MASS_EV",))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["1", "default"]
    r = _run(code, env_extra={"PYUL_AXION_MASS_EV": "1e-21"})
    assert r.stdout.split() == ["0", "env"]


def test_set_axion_mass_bitwise_equal_to_env_import():
    code = ("import Eig_ULDM_packages.units as u\n"
            "import sys\n"
            "if len(sys.argv) > 1: u.set_axion_mass(1e-21)\n"
            "print(repr(u.length_unit), repr(u.mass_unit), repr(u.energy_unit), repr(u.time_unit))\n")
    a = _run(code, env_extra={"PYUL_AXION_MASS_EV": "1e-21"})
    env = dict(os.environ, PYTHONPATH=str(REPO))
    env.pop("PYUL_AXION_MASS_EV", None)
    b = subprocess.run([sys.executable, "-B", "-c", code, "set"], capture_output=True, text=True, env=env)
    assert a.returncode == 0 and b.returncode == 0, (a.stderr, b.stderr)
    assert a.stdout == b.stdout
