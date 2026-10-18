"""Shared helpers for the toolkit tests (synthetic PyUL-like runs, no external data needed)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EV_KG = 1.78266191e-36


def make_cube(n=24, seed=0, side=5.0, width=0.45, shift=(0.11, -0.07, 0.05)):
    """Synthetic soliton-like complex cube with l=1,2 structure and a phase gradient."""
    rng = np.random.default_rng(seed)
    x = (np.arange(n) - (n - 1) / 2) * (side / n)
    X, Y, Z = np.meshgrid(x - shift[0], x - shift[1], x - shift[2], indexing="ij")
    R2 = X ** 2 + Y ** 2 + Z ** 2
    amp = np.exp(-R2 / (2 * width ** 2))
    a1, a2 = rng.normal(size=2) * 0.2
    psi = amp * (1.0 + a1 * Z / width + a2 * (X * Y) / width ** 2 + 0.05j * (X - 2 * Y) / width)
    psi = psi * np.exp(1j * (0.3 * X / width + 0.1 * seed))
    return psi.astype(np.complex128)


def write_run(root, n=24, n_snap=4, save_number=None, rs_record=True, duration=30.0, side=5.0,
              soliton_mass=12.0, axion_ev=1e-21):
    """Write a minimal PyUL run directory: config.uldm, reproducibility.uldm, Outputs/3WfnRS/P3R_#NNN.npy."""
    root = Path(root)
    run = root / "20260101_000000@24"
    (run / "Outputs" / "3WfnRS").mkdir(parents=True)
    save_number = n_snap - 1 if save_number is None else save_number
    config = {
        "PyUL Version": 30.5,
        "Save Options": {"Flags": "Energy NBody 3WfnRS", "Number": save_number, "Format": "npy"},
        "Spatial Resolution": n,
        "Duration": {"Time Duration": duration, "Start Time": 0.0, "Time Units": "Myr"},
        "Simulation Box": {"Box Length": side, "Length Units": "kpc"},
        "ULDM Solitons": {"Condition": [[soliton_mass, [0, 0, 0], [0, 0, 0], 0.0]], "Embedded": [],
                          "Mass Units": "M_solar_masses", "Position Units": "kpc", "Velocity Units": "km/s"},
    }
    repro = {"Integrator Version": 30.5, "Axion Mass": axion_ev * EV_KG, "m22": axion_ev / 1e-22}
    if rs_record:
        repro["SK Grid and Output Settings"] = {
            "Cell Centered Grid": False,
            "Resampled Box": {"resolR": n, "Length_Ratio": 1.0, "Legacy Resample Box": False,
                              "Side (code units)": 0.4120959788338888, "Side (kpc)": side + 4e-15,
                              "Spacing (code units)": 0.4120959788338888 / n, "Spacing (kpc)": side / n},
        }
    (run / "config.uldm").write_text(json.dumps(config, indent=1))
    (run / "reproducibility.uldm").write_text(json.dumps(repro, indent=1))
    for k in range(n_snap):
        np.save(run / "Outputs" / "3WfnRS" / f"P3R_#{k:03d}.npy", make_cube(n, seed=k, side=side))
    return run


@pytest.fixture
def synthetic_run(tmp_path):
    return write_run(tmp_path)


@pytest.fixture(scope="session")
def small_eigen_file(tmp_path_factory):
    """Eigen file (n_grid 128, lmax 2, n_eig 3) for a 12 M_solar_masses soliton, rmax 2.5 kpc, 1e-21 eV."""
    from Eig_ULDM_packages import units
    from Eig_ULDM_packages.pipeline import build_eigen_file
    units.set_axion_mass(1e-21)
    path = tmp_path_factory.mktemp("eig") / "eig_small.h5"
    build_eigen_file(str(path), soliton_mass=12.0, mass_units="M_solar_masses", rmax=2.5, length_units="kpc",
                     n_grid=128, lmax=2, n_eig=3)
    return str(path)
