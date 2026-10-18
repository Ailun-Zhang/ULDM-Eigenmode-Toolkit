# ULDM Eigenmode Toolkit

A Python library for computing the eigenfunctions $f_{nl}(r)$ of a soliton potential and the spectral
coefficients $c_{nlm}(t)$ of ultralight dark matter (ULDM) wavefunctions from
[PyUL_SK](https://github.com/Ailun-Zhang/PyUL_SK) simulations. It is the analysis code of *Stone Skipping
Black Holes in Ultralight Dark Matter Solitons* (A. Zhang, Y. Wang, J. L. Zagorac and R. Easther,
[arXiv:2602.11512](https://arxiv.org/abs/2602.11512)).

The wavefunction around its centre of mass is expanded in the eigenmodes of the soliton potential:

$$\psi(\mathbf{r}, t) = \sum_{n,l,m} c_{nlm}(t)\, f_{nl}(r)\, Y_l^m(\theta, \phi)$$

with radial eigenfunctions $f_{nl}(r)$ of the linearised Schroedinger equation in the soliton potential,
orthonormal complex spherical harmonics $Y_l^m$, and time-dependent coefficients $c_{nlm}(t)$. The chain is

1. eigenfunctions $f_{nl}(r)$ from the soliton density profile (`uldm_eig1`);
2. spherical-harmonic coefficients $a_{lm}(r, t) = \int \psi\, Y_{lm}^*\, d\Omega$ on radial shells of each
   3-D snapshot (`alm_kernel`, historically `functions3` + `compute_coeff.py`);
3. $c_{nlm}(t) = \int f_{nl}(r)\, a_{lm}(r, t)\, r^2\, dr$ (`c_nlm_integrator`, `cnlm_snapshot`);
4. time- and frequency-domain analysis and figures (`cnlm_spectral_analysis`, `paper_figures/`).

Version **0.3.0** (October 2026). Default options reproduce the published numbers; see
[CHANGELOG.md](CHANGELOG.md). The conventions of the published analysis and the known issues of the code
versions behind the paper's figures (with their size and the printed figures they concern) are in
[docs/conventions_and_errata.md](docs/conventions_and_errata.md).

## Contents

1. [Installation](#installation)
2. [Axion mass](#axion-mass)
3. [Three ways to get c_nlm](#three-ways-to-get-c_nlm)
4. [Local pipeline `uldm-cnlm` (laptop, PBS, Slurm)](#local-pipeline-uldm-cnlm-laptop-pbs-slurm)
5. [Python API walkthrough](#python-api-walkthrough)
6. [Historical chain (notebooks)](#historical-chain-notebooks)
7. [File formats](#file-formats)
8. [Conventions and switches](#conventions-and-switches)
9. [Reproducibility](#reproducibility)
10. [Paper figures and data](#paper-figures-and-data)
11. [Module reference](#module-reference)
12. [Environment variables](#environment-variables)
13. [Tests](#tests)
14. [Citation](#citation)
15. [Acknowledgments, licence, changelog](#acknowledgments-licence-changelog)

---

## Installation

Python **3.9 - 3.14** on Windows, macOS and Linux. Tested with Python 3.9.25 (NumPy 1.26.4, SciPy 1.13.1,
pyshtools 4.13.1, the cluster versions used for the paper), Python 3.13 and Python 3.14 (NumPy 2.5.4,
SciPy 1.18.1, pyshtools 4.14.1). IPython is not needed.

```bash
git clone https://github.com/Ailun-Zhang/ULDM-Eigenmode-Toolkit.git
cd ULDM-Eigenmode-Toolkit
python -m pip install -e .            # editable install (recommended)
python -m pip install -e ".[dev]"     # + pytest, nbformat, jupyter, moviepy
```

The distribution is `eig-uldm-packages` (import name `Eig_ULDM_packages`, `Eig_ULDM_packages.__version__ ==
"0.3.0"`); the install also provides the console script `uldm-cnlm`. Dependencies: numpy, scipy,
matplotlib, h5py, tqdm and [pyshtools](https://shtools.github.io/SHTOOLS/) (from PyPI wheels, or
`conda install -c conda-forge pyshtools`). Extras: `anim` (moviepy, for the angular-density videos of
`functions3`) and `dev`. Building needs setuptools >= 77, which supports Python 3.9.

The repository name contains hyphens, so a plain clone is not importable without installation. Copying the
`Eig_ULDM_packages/` folder next to a run (the historical practice) still works, but `pip install -e .` is the
supported route. To use the inline decomposition of PyUL_SK, install the toolkit into the same environment
as PyUL_SK.

```python
import Eig_ULDM_packages
from Eig_ULDM_packages.units import convert_between
from Eig_ULDM_packages.uldm_eig1 import make_all_eigs
from Eig_ULDM_packages.c_nlm_integrator import compute_cnlm
from Eig_ULDM_packages.alm_kernel import compute_alm_cube
print("ULDM Eigenmode Toolkit", Eig_ULDM_packages.__version__)
```

Uninstall with `python -m pip uninstall eig-uldm-packages`.

---

## Axion mass

The code units depend on the axion mass. `Eig_ULDM_packages.units` reads `PYUL_AXION_MASS_EV` **once, at its
first import**; the default is **1e-22 eV**, while the Stone Skipping paper used **1e-21 eV**. Either set the
variable before the first toolkit import or call `units.set_axion_mass(1e-21)` (bitwise the same units).
If the default is used, the first conversion emits one warning.

```python
import os
os.environ["PYUL_AXION_MASS_EV"] = "1e-21"      # before any Eig_ULDM_packages import
from Eig_ULDM_packages import units
print(units.get_axion_mass_ev(), units.axion_mass_source())
print(units.convert(12, 'M_solar_masses', 'm'))   # 12e6 Msun = 170.3149822613627 code masses at 1e-21 eV
```

Every component that reads a PyUL run (`uldm-cnlm`, the inline recorder used by PyUL, `trajectories.load_run`,
the paper-figure scripts) follows one policy, implemented in `Eig_ULDM_packages.pyul_run`:

1. the mass recorded in the run's `reproducibility.uldm` (`"m22"` / `"Axion Mass"`);
2. else an explicit argument (`axion_mass_ev=` / `--axion-mass-ev`); it must agree with a record if both
   exist (`ValueError` otherwise);
3. `PYUL_AXION_MASS_EV` only as a last resort, with a `RuntimeWarning` (the inline recorder inside PyUL never
   uses it: it takes the mass from the run record that PyUL has just written).

---

## Three ways to get c_nlm

| Route | When | What you need |
|---|---|---|
| **Inline in PyUL** (recommended for new runs) | the simulation is still to be run | PyUL_SK >= 30.50: `evolve(..., CenterCalc=True, Cnlm=True)` writes `Outputs/c_nlm.h5` itself; `3WfnRS` need not be saved. See the [PyUL_SK README](https://github.com/Ailun-Zhang/PyUL_SK#inline-eigenmode-decomposition-c_nlm). |
| **Local pipeline `uldm-cnlm`** | existing runs with saved `Outputs/3WfnRS/P3R_#NNN.npy` | one command per run, on a laptop or one cluster node; resumable |
| **Historical chain** (notebooks + `compute_coeff.py`) | regenerating the paper's intermediate JSON files | `Main.ipynb` -> `compute_coeff.py` (per-shell JSON, PBS fan-out) -> `create_sh_hdf5` -> `f_nl_python.ipynb` -> `compute_c_nlm.ipynb` |

With the same kernel parameters, all three give the same numbers (bit for bit within one process; see
[Reproducibility](#reproducibility)) and write the same `c_nlm.h5` layout. The default shell count agrees for a
256^3 resampled cube (257 shells, as in the paper). For other `resolR` the inline recorder keeps its fixed
default `CnlmNShells=257` while `uldm-cnlm` uses `resolR + 1`, so pass `--from-inline` (which copies the inline
settings) or `--n-shells` when comparing the two.

Inline, the PyUL call is

```python
PyUL.evolve(save_path, run, CenterCalc=True, Length_Ratio=1, resolR=256, CellCenteredGrid=True, Cnlm=True,
            OverwriteOutputs=True)
```

and a run that died before the end is completed from its per-save checkpoints with

```bash
uldm-cnlm path/to/run --from-inline
```

`--from-inline` refuses an eigen file whose SHA-256 differs from the one the run recorded (pass `--eigen-file`
to choose a file, or `--force` to continue with the changed one). The checkpoints stay usable after a toolkit
update that does not change the a_lm kernel, and runs saved with `Save Options Number = -1` (every step) are
supported through `Outputs/snapshot_times_Myr.npy`.

---

## Local pipeline `uldm-cnlm` (laptop, PBS, Slurm)

`uldm-cnlm RUN_DIR` (or `python -m Eig_ULDM_packages.pipeline RUN_DIR`) computes a_lm and c_nlm for one PyUL
run without PBS. `RUN_DIR` is the run folder (the one with `config.uldm`), its `Outputs/`, or a save_path
with exactly one run. Input: `Outputs/3WfnRS/P3R_#NNN.npy`, i.e. PyUL must have run with `CenterCalc=True`
and the `3WfnRS` save flag; only `.npy` saves are read.

```bash
uldm-cnlm path/to/run --build-eigen                          # paper defaults; builds the eigen file too
uldm-cnlm path/to/run --eigen-file eigs.h5 --alm-out a_lm.h5 --workers 8
uldm-cnlm path/to/run --rs-side 5.0 --rs-resol 256 --build-eigen   # runs of PyUL <= 30.40 (geometry not recorded)
uldm-cnlm path/to/run --build-eigen --snapshots 0:500:10 --dry-run  # print the resolved parameters and exit
uldm-cnlm --help
```

Each line is an independent example. The pipeline

1. reads the run metadata: duration, save number, axion mass, soliton mass and, for PyUL_SK >= 30.50, the
   resampled-box geometry from `reproducibility.uldm` (`"SK Grid and Output Settings"` -> `"Resampled Box"`);
2. optionally builds the eigen file with the paper defaults (`--build-eigen`);
3. computes a_lm(r) of every requested snapshot with the one-pass kernel and writes one checkpoint per
   snapshot (an interrupted run resumes where it stopped; checkpoints made with other kernel parameters, or
   from snapshot files that changed since, are refused);
4. writes `c_nlm.h5` and optionally the a_lm file, both with provenance attributes.

**Defaults (= the paper analysis).** Snapshots 0 ... save_number; `n_shells = resolR + 1` (257 for a 256^3
resampled cube) from `r[0] = 1e-5` to half the resampled box; `l_max` = the eigen file's lmax (10);
`num_points = 300`; `angular_grid = legacy`; centre = `|psi|^2`-weighted COM of the cube. The eigen file
(`--build-eigen`) is built from the single soliton mass in `config.uldm` with `n_grid = 5120` (including
r = 0), lmax 10, `n_eig = 25`, `r_max` = half the resampled box, `maxiter = 2e6`, `tol = 1e-10` and the legacy
boundary, and is named `eigs_mass_<mass><units>_Eig_25_lmax_10_nig_5120.h5`.

**Resampled-box geometry.** PyUL <= 30.40 does not record `Length_Ratio`, `resolR` or `CellCenteredGrid`;
give `--rs-side <side> --rs-resol <N>` (or `--rs-dx`) in the run's length units. With `CellCenteredGrid=True`,
PyUL 30.38 - 30.40 resampled onto a side of `(L - L/N) * Length_Ratio`, not `L * Length_Ratio`; the
historical scripts assumed `L * Length_Ratio`, a 1/N radial mislabel (0.78 / 0.52 / 0.39 % at
N = 128 / 192 / 256). This concerns the c_nlm of printed Figs. 2 and 12 of the paper (P_dip 0.6 - 2.4 % and
|c_000|^2/M 1.2 - 2.4 % too high; frequencies unaffected); see
[errata E4](docs/conventions_and_errata.md#e4-pyul-3038--3040-resampled-box-a_lm-radii-of-fig-2-and-fig-12).

**Outputs** go to `<run>/Outputs/cnlm_pipeline/` (`--out-dir`): `c_nlm.h5`, the optional a_lm file
(`--alm-out`, relative names are inside the output folder), `alm_checkpoints/ALM_#NNN.npz`,
`pipeline_params.json`, `pipeline_summary.json`, and the eigen file when `--build-eigen` is used. Existing
final outputs are never overwritten without `--force`.

**Time axis.** PyUL's real save times `Outputs/snapshot_times_Myr.npy` when the file exists (PyUL_SK >=
30.50), else `linspace(0, duration, save_number + 1)[snapshots]` in Myr; the choice is recorded in the
`time_axis` attribute.

**Threads.** `--workers`, else `ULDM_NUM_WORKERS` / `PBS_NCPUS` / `NCPUS` / `SLURM_CPUS_PER_TASK` /
`OMP_NUM_THREADS`, else the CPU affinity. Measured cost for one 256^3 cube with 257 shells, N = 300, lmax 10
(16-core workstation): 18.5 s on 1 thread, 5.1 s on 8, 4.0 s on 16; about 4 s on 56 cluster cores. The
historical per-(snapshot, shell) kernel needs about 1.5 s per shell, i.e. about 6.6 min per snapshot per
core.

### Options

| Option | Default | Meaning |
|---|---|---|
| `run_dir` | (required) | run folder, its `Outputs/`, or a save_path with one run |
| `--input-prefix` | `3WfnRS/P3R_#` | snapshot prefix relative to `Outputs` |
| `--snapshots` | 0 ... save_number | `a:b` (inclusive), `a:b:step`, `i,j,k` or `all` |
| `--skip-missing` | off | drop missing snapshots instead of stopping |
| `--axion-mass-ev` | from the record | axion mass for runs that do not record it |
| `--rs-side`, `--rs-resol`, `--rs-dx` | from the record | resampled-cube geometry (run's length units) |
| `--length-units` | run's (kpc) | length units of `--rs-*` and the shells |
| `--n-shells` | `resolR + 1` | number of shells |
| `--r-max` | half the resampled box | outer shell radius |
| `--r0` | `1e-5` | radius of the first shell |
| `--shells-file` | - | `.npy` with explicit shell radii |
| `--l-max` | eigen file's lmax | maximum degree |
| `--num-points` | 300 | N of the N x 2N angular grid |
| `--angular-grid` | `legacy` | `legacy` (paper) or `dh` (Driscoll-Healy) |
| `--workers` | see above | threads |
| `--eigen-file` | - | existing eigen HDF5 |
| `--build-eigen` | off | build the eigen file with the paper defaults |
| `--soliton-mass`, `--soliton-mass-units` | from `config.uldm` | soliton mass for `--build-eigen` |
| `--eigen-rmax`, `--eigen-n-grid`, `--eigen-lmax`, `--eigen-n-eig`, `--eigen-maxiter` | shell r_max, 5120, 10, 25, 2e6 | eigen-file parameters |
| `--eigen-outer-boundary` | `legacy` | `legacy` or `rmax` (see [switches](#conventions-and-switches)) |
| `--soliton-mass0` | file value 3.883 | override the bundled profile's rounded `mass0` (e.g. 3.8827652755822006) |
| `--out-dir` | `<run>/Outputs/cnlm_pipeline` | output folder |
| `--cnlm-out` | `c_nlm.h5` | c_nlm file name or path |
| `--alm-out` | - | also write the a_lm HDF5 (`create_sh_hdf5` layout) |
| `--no-cnlm` | off | only a_lm (checkpoints / `--alm-out`) |
| `--force` | off | recompute checkpoints and overwrite outputs |
| `--dry-run` | off | resolve and print all parameters, then exit |
| `--from-inline` | off | take parameters and checkpoints from PyUL's inline decomposition (`Outputs/Cnlm/cnlm_inline.json`) and assemble or complete `c_nlm.h5`; snapshots without a checkpoint are computed from their `3WfnRS` file |

### Cluster templates

[examples/pbs/run_cnlm_pipeline.pbs](examples/pbs/run_cnlm_pipeline.pbs) and
[examples/slurm/run_cnlm_pipeline.slurm](examples/slurm/run_cnlm_pipeline.slurm) run the pipeline for one run
on one node (`set -euo pipefail`, no host pinning, a unique log per job, `ULDM_NUM_WORKERS` from `NCPUS` or
`SLURM_CPUS_PER_TASK`). Edit `RUN_DIR`, `PYTHON`, `PIPELINE_ARGS` and the resource request, then

```bash
qsub examples/pbs/run_cnlm_pipeline.pbs
sbatch examples/slurm/run_cnlm_pipeline.slurm
```

Re-submitting the same job resumes from the checkpoints. `submit_all_jobs_example.sh` is the **PBS**
submitter of the historical per-shell stage (`compute_coeff<i>.py`); the pipeline replaces it.

---

## Python API walkthrough

The same steps as the pipeline, for one snapshot. The example uses the paper's settings: a 12e6 Msun soliton,
a 256^3 resampled cube of side 5 kpc, eigenfunctions on [0, 2.5 kpc].

```python
import os
os.environ["PYUL_AXION_MASS_EV"] = "1e-21"                    # before the first toolkit import
import numpy as np
from importlib import resources
from Eig_ULDM_packages import units
from Eig_ULDM_packages.uldm_eig1 import (load_soliton_file, soliton_density_function,
                                         compute_potential, make_all_eigs, save_eigs_to_h5)
from Eig_ULDM_packages.c_nlm_integrator import load_eigenfunctions
from Eig_ULDM_packages.alm_kernel import compute_alm_cube
from Eig_ULDM_packages.cnlm_snapshot import legacy_alm_radial_grid, cnlm_from_alm_snapshot

# 1. Eigenfunctions f_nl(r) (as f_nl_python.ipynb and uldm-cnlm --build-eigen).
sol = load_soliton_file(str(resources.files("Eig_ULDM_packages") / "soliton.h5"))
m_code = units.convert(12, 'M_solar_masses', 'm')                 # soliton mass in code units
r_grid = np.linspace(0, units.convert(2.5, 'kpc', 'l'), 5120)     # code units, includes r = 0
phi = compute_potential(r_grid, soliton_density_function(m_code, sol))
eigs = make_all_eigs(r_grid, phi, lmax=10, n_eig=25, maxiter=2_000_000, tol=1e-10)
save_eigs_to_h5(eigs, "eigs_mass_170_Eig_25_lmax_10_nig_5120.h5")
r_f, f_dict = load_eigenfunctions("eigs_mass_170_Eig_25_lmax_10_nig_5120.h5")

# 2. a_lm(r) of one resampled snapshot (PyUL Outputs/3WfnRS/P3R_#NNN.npy).
psi = np.load("path/to/run/Outputs/3WfnRS/P3R_#000.npy")
dx_kpc = 5.0 / 256                                                # resampled side / resolR
r_kpc = np.linspace(0, 2.5, 257)
r_kpc[0] = 1e-5                                                   # the paper's 257 shells
alm = compute_alm_cube(psi, dx_kpc, r_kpc, l_max=10, num_points=300, angular_grid='legacy')

# 3. c_nlm of that snapshot, computed exactly as the historical compute_cnlm chain.
r_al = legacy_alm_radial_grid(2.5, 257, r_f[-1], length_units='kpc')   # shells in code units
c = cnlm_from_alm_snapshot(alm, r_al, r_f, f_dict, num_points_roundtrip=300)
print("|c_000|^2 / M =", abs(c[(0, 0, 0)]) ** 2 / m_code)
```

`compute_alm_cube` returns an `(n_r, (l_max+1)^2)` complex array with columns l = 0 ... l_max, m = -l ... l;
`dx` and the shell radii must be in the same unit, and the centre is the `|psi|^2`-weighted COM of the cube
(`center='com'`). For many snapshots build one `AlmKernel(r_values, l_max, num_points, angular_grid,
workers)` and call `kernel.compute(psi, dx, shell_block=32)` (bounded memory, identical values).
`num_points_roundtrip=300` applies the a_lm storage round trip of the historical files (needed for bitwise
equality).

Working with files (here the output of `uldm-cnlm path/to/run --build-eigen --alm-out a_lm.h5`):

```python
import numpy as np
from Eig_ULDM_packages.c_nlm_integrator import load_eigenfunctions, load_a_lm, compute_cnlm, load_cnlm_from_h5

out = "path/to/run/Outputs/cnlm_pipeline"
r_f, f_dict = load_eigenfunctions(f"{out}/eigs_mass_12M_solar_masses_Eig_25_lmax_10_nig_5120.h5")
A = load_a_lm(f"{out}/a_lm.h5", None)          # r (code units) and t (Myr) from the file's attributes
t, c = compute_cnlm(r_f, f_dict, A["r"], A["t"], A["a_lm"])
t_file, c_file, attrs = load_cnlm_from_h5(f"{out}/c_nlm.h5")
print(A["time_source"], len(t), max(np.max(np.abs(c[k] - c_file[k])) for k in c))
```

Spectral analysis of one mode (as printed Fig. 12 of the paper):

```python
from pathlib import Path
from Eig_ULDM_packages import units
from Eig_ULDM_packages.c_nlm_integrator import load_cnlm_from_h5
from Eig_ULDM_packages.cnlm_spectral_analysis import (analyze_cnlm_spectrum, print_spectral_summary,
                                                      plot_4panel_eda, export_prd_figure)

t, c, attrs = load_cnlm_from_h5("path/to/run/Outputs/c_nlm.h5")
M = units.convert(12, 'M_solar_masses', 'm')
res = analyze_cnlm_spectrum(t, c[(1, 1, -1)], (1, 1, -1), normalization=M, peak_rel_threshold=0.01)
print_spectral_summary(res)
plot_4panel_eda(res).savefig("cnlm_n1_l1_m-1_eda.png", dpi=150)
export_prd_figure(res, save_path=Path("cnlm_n1_l1_m-1_prd"), time_xlim=(0, 3000))   # .eps and .pdf
```

Black-hole trajectories without PyUL:

```python
from Eig_ULDM_packages.trajectories import load_run

run = load_run("path/to/run")                  # run folder, save_path with one run, or a .npz bundle
print(run.summary())
r_origin = run.distance_from_origin_pc(body=0)      # |x_BH| in pc
r_com = run.distance_from_uldm_com_pc(body=0)       # |x_BH - x_ULDCOM| in pc
print(run.times_myr[:3], r_origin[:3], r_com[:3])
```

`python -m Eig_ULDM_packages.trajectories info RUN` prints the same summary; `... export RUN out.npz` writes a
single-file trajectory bundle.

---

## Historical chain (notebooks)

The paper's coefficients were made with this chain, which is still supported:

1. **`Main.ipynb` Step 1** builds the shell radii `r_0_values.npy` (snapshot range 0 ... save_number from
   `config.uldm`; "resolution" and "box_length" are those of the resampled `3WfnRS` cube) and the
   `compute_coeff<i>.py` scripts from the template.
2. **`compute_coeff.py`** (run from the run's `Outputs/`, or with `--outputs-dir`) writes one JSON file of
   a_lm per (snapshot, shell). Options `--start/--end`, `--rs-length-kpc/--rs-resol` (the 3WfnRS cube, not
   the simulation box), `--l-max`, `--num-points`, `--angular-grid`, `--workers`, `--cores-per-r0`
   (default 4), `--skip-existing`; the same settings as `ULDM_*` environment variables. A sidecar
   `alm_coefficients/<folder>/kernel_params_<prefix>lmax_<l>.json` records dx, the RS resolution, `num_points`,
   the angular grid and `l_max`; a rerun with other values into the same tree (with or without
   `--skip-existing`) is refused unless `--no-params-check` is given. It exits non-zero
   when any shell fails and writes a summary JSON. `submit_all_jobs_example.sh` fans it out over PBS jobs.
3. **`Main.ipynb` Step 3** merges the JSON files with `create_sh_hdf5` into the a_lm HDF5 file. Missing or
   corrupt JSON files raise an error; `allow_missing=True` restores the old zero-filling and records what was
   missing.
4. **`f_nl_python.ipynb`** Steps 1 - 5 build the eigen file (also: wavefunction reconstruction with
   `build_initial_wavefunction`, `deboost_wavefunction`, `compute_mode_fractions`).
5. **`compute_c_nlm.ipynb`** loads both files (`load_eigenfunctions`, `load_a_lm`), checks the radial grids
   (`rescale_r_to_eigen_rmax`), computes `compute_cnlm`, saves `save_cnlm_to_h5`, and makes the dense
   display series with `resample_cnlm_time(..., 'nearest-savgol')` (about 2.9 GB per run; piecewise-constant
   phase; for plots only, not for spectra).

```bash
python compute_coeff.py --help
```

---

## File formats

All layouts are unchanged from the historical files; 0.3.0 tools add provenance only as **root attributes**
(no new groups or datasets; arrays longer than 4096 elements are split into numbered attribute chunks).

**Eigen file** (`save_eigs_to_h5`, `f_nl_python.ipynb`, `--build-eigen`, PyUL's automatic file):
`/ell/<l>/r` (n_grid), `/ell/<l>/E` (n_eig, ascending, n is 0-based) and `/ell/<l>/f` (n_grid x n_eig), all in
code units; normalisation `int_0^rmax f_nl^2 r^2 dr = 1` (no 4 pi). There is no root `r_grid` dataset.
Built by 0.3.0 tools it carries attributes such as `soliton_mass`, `rmax`, `n_grid`, `lmax`, `n_eig`,
`includes_r0`, `maxiter`, `tol`, `outer_boundary`, `axion_mass_ev`, `soliton_mass0`.

**a_lm file** (`create_sh_hdf5`, `write_alm_h5`, `--alm-out`, `CnlmSaveAlm`): `/massArr` (n_t, constant,
unused); `/alm_%06d/counts` (n_r x (l_max+1)^2, the constant `num_points * 2 * num_points`); `/alm_%06d/ylm`
= a_lm * counts (complex128), columns l = 0 ... l_max, m = -l ... l. Groups are numbered from 0 for the first
snapshot in the file. `ylm / counts` is not an exact identity (last-bit changes); readers apply it, as the
historical chain did.

**c_nlm file** (`save_cnlm_to_h5`, `write_cnlm_h5`, the pipeline, PyUL inline): root `t` (float64, Myr) and
`n/<n>/l/<l>/m_<m>` (complex128 series). Raw c_nlm in sqrt(code mass) units: `sum |c|^2` = mass in code
units, a pure soliton has `c_000 = sqrt(M)`, and the paper plots `|c|^2 / (M (1 + eps^2))`.

**Root provenance attributes** written by 0.3.0 (historical files have none): shells `r_values`, `r_units`,
`r_values_code`; times `snapshot_indices`, `times`, `time_units`, `time_axis`; kernel `l_max`, `num_points`,
`angular_grid`, `center`; resampled cube `rs_side`, `rs_resol`, `rs_dx`, `rs_source`; `axion_mass_ev`,
`eigen_file_sha256`; `version_*`, `producer`, `created_utc`; PyUL inline files add `source='pyul-inline'`,
`pyul_version`, `pyul_core_sha256`, `rs_legacy_resample_box`, `rs_cell_centered_grid`, `stopped_early`.
`load_a_lm` uses `r_values_code` and `times` when present; `load_cnlm_from_h5` returns the attributes as its
third value.

**Spherical harmonics.** `pyshtools.expand.SHExpandDHC(norm=4, sampling=2, csphase=-1)`: orthonormal
complex Y_lm with the Condon-Shortley phase, `a_lm(r) = int psi Y_lm^* dOmega`.

---

## Conventions and switches

Default options reproduce the published numbers. Options that change numbers are opt-in (policy B in the
changelog). [docs/conventions_and_errata.md](docs/conventions_and_errata.md) quantifies each legacy convention
and lists the printed figures it concerns.

| Switch | Default (paper) | Alternative | Effect |
|---|---|---|---|
| angular grid: `angular_grid` (`compute_alm_cube`, `AlmKernel`, `functions3.process_file`, `compute_sh_coeffs`), `--angular-grid`, `CnlmAngularGrid` | `'legacy'`: `theta = linspace(0, pi, N)`, `phi = linspace(0, 2 pi, 2N)` (both endpoints; not the Driscoll-Healy grid) | `'dh'`: `theta_i = pi i / N`, `phi_j = pi j / N` | At N = 300 the legacy grid gives Y00 exactly; for l >= 1 amplitude errors of 0.1 - 0.4 %, a phase offset `m pi / (2N - 1)` (0.005 rad at m = 1, 0.052 rad at m = 10) and 0.3 - 4 % leakage into other (l', m'). `'dh'` is exact to 1e-15. |
| eigen-solver outer boundary: `outer_boundary` (`solve_h`, `make_all_eigs`), `--eigen-outer-boundary`, `CnlmEigenParams` | `'legacy'`: the last grid point is an unknown, so the Dirichlet wall is effectively at `r_max + h` | `'rmax'`: `u(r_max) = 0` exactly | `'rmax'` converges at second order to the fixed-domain spectrum. |
| soliton `mass0`: `load_soliton_file(path, mass0=None)`, `--soliton-mass0` | the bundled `soliton.h5` value 3.883 (rounded) | 3.8827652755822006 (exact) | The rounded value makes the scaled soliton about 6e-5 too light; PyUL's standard solitons use the exact value. |
| time axis | PyUL save times `start + k * duration / save_number` (Myr); `load_a_lm(time_source='auto')` uses a file's `times` attribute, else `linspace(0, final_time, n)` | `times=`, `snapshot_indices=` + `save_interval=`, `time_source='legacy'`/`'file'` | Historical a_lm files always start at t = 0 for their first group. Some old analysis notebooks used `k * duration / (save_number + 1)` for trajectories (printed Figs. 4, 8, 9 and the digitised sources of Figs. 10, 11; `trajectories.load_run(legacy_time_axis=True)` reproduces it). |
| `resample_cnlm_time(method=...)` | `'pchip'` | `'nearest'` (pure nearest-neighbour since 0.3.0), `'nearest-savgol'` (the paper's dense display files of printed Figs. 5 and 12) | display only |
| `functions3.process_file(raise_on_error=True)` | re-raise after logging | `False`: old print-and-continue | |
| `create_sh_hdf5(allow_missing=False)` | missing JSON is an error | `True`: zero-fill and record `missing_*` attributes | |
| `compute_potential_from_grid(cell_centered=True)` | cell-centred grid | `False` for PyUL's node grid | warns when it infers `dx = 2 r_max / N` |

Other behaviour changes in 0.3.0 users may notice: `from functions3 import *` no longer exports `plt`,
`ImageSequenceClip`, `FormatStrFormatter` or `FixedLocator`; file names such as
`PyUL_psi_alm_0.02_mass170_256.h5` are no longer parsed as r_max = 0.02 (pass `r_al`, or put an explicit
`rmax<value>` token in the name); `convert_back(v, 'c', 'v')` is fixed (it was inverted); every process pool of the toolkit uses
`'fork'` on Linux and `'spawn'` elsewhere (`ULDM_MP_START_METHOD` overrides).

---

## Reproducibility

* With default options, the a_lm and c_nlm values of the historical chain
  (`process_file` -> JSON -> `create_sh_hdf5` -> `load_a_lm` -> `compute_cnlm` -> `save_cnlm_to_h5`) are
  **bitwise identical** to version 0.2.0 (the version used for the paper analysis) when run in the same
  process, and the new pipeline and PyUL's inline recorder are bitwise identical to that chain within one
  process (SciPy 1.13.1 and 1.18.1, 1 - 64 threads).
* pyshtools' `SHExpandDHC` is deterministic within a process but **not across processes** (last-bit
  differences of about 1e-17 absolute, consistent with FFT planning). Two runs of the historical
  multi-process chain therefore already differ at about 3e-16 relative in a_lm and about 1e-16 of max|c| in
  c_nlm; the new code agrees with the historical chain to the same level (<= 3.2e-14 relative per mode in
  the inline tests).
* Cluster validation (PBS job on the production cluster, Python 3.9.18, NumPy 1.26.4, SciPy 1.13.1, pyshtools
  4.13.1, 56 threads): on real paper snapshots of PyUL 26.32 and 30.40 runs (256^3 and 128^3 at three times,
  192^3 at two) the one-pass kernel reproduces the historical a_lm JSON files to <= 4.2e-16 relative and the
  historical c_nlm files to <= 1.4e-16 of max|c| over all 3025 modes, at about 4 s per snapshot; the `r`, `f`
  and `E` datasets of the eigen file built automatically for a 128^3 PyUL run are bitwise identical to those of
  the paper's eigen file (sha256 `2bb42868...`). The file's own SHA-256 differs, because 0.3.0 adds provenance
  root attributes.
* The eigen solver (`scipy.sparse.linalg.eigsh`, `tol = 1e-10`) is bit-reproducible across processes with the
  cluster versions (Python 3.9, SciPy 1.13.1), but not in every environment: with Python 3.13 / SciPy 1.18.1
  on Windows, repeated builds of the same eigen file differed by up to 1e-9 in E (about 1e-12 relative) and
  1e-10 in f (about 4e-14 of max|f|). Keep the eigen file of a run (its SHA-256 is recorded in every c_nlm
  file) rather than rebuilding it when bitwise comparisons matter.

---

## Paper figures and data

[`paper_figures/`](paper_figures/README.md) regenerates all 13 printed figures of the paper with 12 scripts
named after the printed figure numbers (`fig01_*.py` ... `fig13_*.py`; `fig06_fig07_cnlm_single_modes.py`
makes the two single-mode figures 6 and 7) and `make_all_figures.py`. They use the small data products (per-save
black-hole trajectories or single-file trajectory bundles, the `c_nlm` files and the eigenbasis); the 3-D
wavefunctions are not needed. Every script writes a `<basename>.provenance.json` with the input files and
their SHA-256, PyUL versions, axion masses and time-axis conventions. See
[paper_figures/README.md](paper_figures/README.md) for the data layout, the conventions of each figure and the
validation against the published figures.

```bash
python paper_figures/make_all_figures.py --help
python paper_figures/make_all_figures.py --data-root path/to/StoneSkipping_PRD_data_v1 --out-dir figures_out
```

Relative `--data-root`, `--out-dir` and `--config` paths are relative to the current working directory.

The data deposit *Data for "Stone Skipping Black Holes in Ultralight Dark Matter Solitons"* (version 1,
Zenodo, CC BY 4.0) holds the trajectory bundles, c_nlm and eigenbasis files and the provenance tables of the
runs, plus a copy of `paper_figures/`; its README starts with a step-by-step guide to reproduce the figures
and the simulations in your own environment. The 3-D wavefunction outputs are not deposited; they are
available from the corresponding author on reasonable request.

---

## Module reference

| Module | Content |
|---|---|
| `units` | `convert`, `convert_back`, `convert_between` (PyUL unit strings), `set_axion_mass`, `get_axion_mass_ev`, `axion_mass_source` |
| `uldm_eig1` | soliton profile, potential and eigen solver: `load_soliton_file`, `soliton_density_function`, `compute_potential`, `compute_potential_from_grid`, `make_all_eigs`, `solve_h`, `save_eigs_to_h5`, `compare_with_reference` |
| `alm_kernel` | one-pass a_lm(r) of a cube: `compute_alm_cube`, `AlmKernel`, `angular_grid_1d`, `lm_list` |
| `cnlm_snapshot` | per-snapshot c_nlm and writers: `CnlmProjector`, `cnlm_from_alm_snapshot`, `alm_counts_roundtrip`, `legacy_alm_radial_grid`, `write_cnlm_h5`, `write_alm_h5`, checkpoint helpers |
| `c_nlm_integrator` | file-level chain: `load_eigenfunctions`, `load_a_lm`, `compute_cnlm`, `resample_cnlm_time`, `save_cnlm_to_h5`, `load_cnlm_from_h5`, `check_rmax_consistency`, `rescale_r_to_eigen_rmax` |
| `alm_utils` | `create_sh_hdf5` (merge the per-shell JSON files), `reorganize_coeffs`, `analyze_h5_structure` |
| `functions3` | historical per-shell kernel `process_file`, `compute_sh_coeffs`; plotting and video helpers |
| `pipeline` | the `uldm-cnlm` command (`main`, `run_pipeline`, `build_eigen_file`) |
| `cnlm_inline` | recorder behind PyUL_SK's `evolve(Cnlm=True)` (`CnlmInlineSettings`, `InlineCnlmRecorder`) |
| `pyul_run` | the single PyUL run loader: `PyULRun`, `find_run_dir`, `read_run_metadata`, unit system, time axis, axion-mass policy, `rs_geometry` |
| `trajectories` | trajectories, energies and time axis of a run or bundle: `load_run`, `export_bundle`, `load_bundle`, `PyULRun` (with `distance_from_origin_pc`, `distance_from_uldm_com_pc`, ...) |
| `cnlm_spectral_analysis` | `analyze_cnlm_spectrum`, `print_spectral_summary`, `plot_4panel_eda`, `export_prd_figure`, `plot_cnlm_prd_single`, `plot_cnlm_multimode`, `plot_cnlm_comparison` |
| `cnlm_postprocess` | `prepare_epsilon_nlm` |
| `wavefunction_tools` | initial conditions from eigenmode superpositions: `build_initial_wavefunction`, `deboost_wavefunction`, `compute_mode_fractions` |
| `fnl_plotting` | `plot_eigenfunction_panel` |

Signatures of the main functions (0.3.0):

```text
uldm_eig1.load_soliton_file(h5_path, mass0=None) -> {'ymax', 'psi', 'alpha_func'}
uldm_eig1.soliton_density_function(m, sol) -> rho(r)
uldm_eig1.compute_potential(r_grid, rho_func, *, zero_at_rmax=False) -> phi
uldm_eig1.make_all_eigs(r_grid, phi, lmax, n_eig, shift=None, maxiter=None, tol=1e-10, outer_boundary='legacy') -> {'ell/<l>': {'r', 'E', 'f'}}
uldm_eig1.save_eigs_to_h5(data, h5_path, metadata=None)
alm_kernel.compute_alm_cube(psi_cube, dx, r_values, l_max, num_points=300, angular_grid='legacy', center='com', workers=None, return_raw=False, chunk_points=None) -> (n_r, (l_max+1)**2)
alm_kernel.AlmKernel(r_values, l_max, num_points=300, angular_grid='legacy', workers=None, chunk_points=None).compute(psi_cube, dx, center='com', return_raw=False, prefiltered=None, return_center=False, shell_block=None)
cnlm_snapshot.legacy_alm_radial_grid(r_max, n_r, r_max_eig, *, r0=1e-05, length_units='kpc', rtol=1e-06) -> r_al (code units)
cnlm_snapshot.cnlm_from_alm_snapshot(alm, r_al, r_f, f_dict, *, l_max=None, r_interp_kind='cubic', num_points_roundtrip=None) -> {(n, l, m): complex}
cnlm_snapshot.write_cnlm_h5(path, t, c_dict, *, metadata=None, overwrite=False)
cnlm_snapshot.write_alm_h5(path, alm_snapshots, num_points, *, mass_value=170.3149822613627, metadata=None, overwrite=False)
alm_utils.create_sh_hdf5(output_path, start_index, end_index, n_r, l_max, num_points, r_values, file_prefix='3WfnRS/P3R_#', mass_value=170.3149822613627, n_workers=None, *, allow_missing=False, times=None, time_units=None, r_units=None, angular_grid=None, metadata=None)
c_nlm_integrator.load_eigenfunctions(eigs_path, l_subset=None, n_subset=None) -> (r_f, {(l, n): f})
c_nlm_integrator.load_a_lm(alm_path, r_al, r_max_override=None, final_time=1.0, l_max=None, start_index=None, end_index=None, verbose=False, *, times=None, snapshot_indices=None, save_interval=None, time_source='auto') -> {'r', 't', 'a_lm', 'snapshot_indices', 'time_source', 'attrs'}
c_nlm_integrator.compute_cnlm(r_f, f_dict, r_al, t_al, a_lm, *, r_interp_kind='cubic', time_interp=None, t_out=None, grid_choice='eigen', r_manual=None, num_workers=1, progress=False) -> (t, {(n, l, m): series})
c_nlm_integrator.resample_cnlm_time(c_dict, t_in, t_out, method='pchip')
c_nlm_integrator.save_cnlm_to_h5(c_dict, t_array, output_path, metadata=None)
c_nlm_integrator.load_cnlm_from_h5(path) -> (t, {(n, l, m): series}, attrs)
cnlm_spectral_analysis.analyze_cnlm_spectrum(t_data, c_data, key, analysis_mode='complex', normalization=None, apply_hann_window=True, detrend_mean=True, peak_rel_threshold=0.05, top_k_peaks=6)
cnlm_spectral_analysis.plot_4panel_eda(result, figsize=(13, 9))
cnlm_spectral_analysis.export_prd_figure(result, save_path=None, figsize=(7.6, 3.1), time_xlim=None, save_pdf=True, time_unit='Myr')
cnlm_spectral_analysis.plot_cnlm_prd_single(t, c, key, normalization=1.0, save_path=None, xlim=(0.0, 3000.0), figsize=(3.5, 2.6), color='red', time_unit='Myr')
trajectories.load_run(path, *, axion_mass_ev=None, legacy_time_axis=False, load_energies=True)
```

---

## Environment variables

| Variable | Effect |
|---|---|
| `PYUL_AXION_MASS_EV` | axion mass in eV for `units` (read at first import; default 1e-22); last resort for run readers |
| `ULDM_NUM_WORKERS` | thread / worker count of the pipeline and kernel (then `PBS_NCPUS`, `NCPUS`, `SLURM_CPUS_PER_TASK`, `OMP_NUM_THREADS`, CPU affinity) |
| `ULDM_MP_START_METHOD` | multiprocessing start method (`fork` on Linux, `spawn` elsewhere by default) |
| `ULDM_START_INDEX`, `ULDM_END_INDEX`, `ULDM_LMAX`, `ULDM_RS_LENGTH_KPC`, `ULDM_RS_RESOL`, `ULDM_FILE_PREFIX`, `ULDM_NUM_POINTS`, `ULDM_ANGULAR_GRID`, `ULDM_CORES_PER_R0` | settings of `compute_coeff.py` |
| `ULDM_EQUIV_RUN_DIR` | tests: a PyUL run with `Outputs/3WfnRS/P3R_#000..003.npy` for the real-run equivalence test |
| `ULDM_PAPER_DATA_ROOT` | tests: the paper data root for the figure data checks |

---

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

About 1 minute. The suite covers the units and compatibility shims, the eigen solver (including the `'rmax'`
boundary on the analytic sphere), the angular grids, the a_lm kernel (bitwise equality with `process_file`
for 1 - 64 threads and any chunking), the per-snapshot projector, the pipeline (bitwise equality with the
historical chain, resume, refusal of mismatched checkpoints), the inline recorder, the trajectory loader and
the paper-figure scripts. `tests/test_equivalence_real.py` runs when `ULDM_EQUIV_RUN_DIR` is set, and the
figure data checks run when the paper data are available (`ULDM_PAPER_DATA_ROOT`). The suite runs in the
three test environments (Python 3.9 with the cluster versions, 3.13 and 3.14); on a machine without the paper
data the data checks are skipped.

---

## Citation

If you use this toolkit, please cite the paper and the software release:

```bibtex
@misc{zhang2026stoneskipping,
  author        = {Zhang, Alan and Wang, Yourong and Zagorac, J. Luna and Easther, Richard},
  title         = {Stone Skipping Black Holes in Ultralight Dark Matter Solitons},
  year          = {2026},
  eprint        = {2602.11512},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO},
  doi           = {10.48550/arXiv.2602.11512},
  url           = {https://arxiv.org/abs/2602.11512}
}

@software{uldm_eigenmode_toolkit,
  author  = {Zhang, Alan},
  title   = {ULDM Eigenmode Toolkit},
  version = {0.3.0},
  year    = {2026},
  url     = {https://github.com/Ailun-Zhang/ULDM-Eigenmode-Toolkit}
}
```

Each GitHub release of this repository is archived on Zenodo, which mints a DOI for it; please add the DOI
of the release you used (shown on its Zenodo record) to the software entry.

The eigenmode decomposition builds on

```bibtex
@article{zagorac2022schrodinger,
  author  = {Zagorac, J. L. and Sands, I. and Padmanabhan, N. and Easther, R.},
  title   = {Schr{\"o}dinger-Poisson solitons: Perturbation theory},
  journal = {Physical Review D},
  volume  = {105},
  pages   = {103506},
  year    = {2022},
  doi     = {10.1103/PhysRevD.105.103506}
}
```

Simulations come from [PyUL_SK](https://github.com/Ailun-Zhang/PyUL_SK) (cite as described in its README,
together with PyUltraLight and PyUltraLight 2). Spherical-harmonic transforms use
[pyshtools](https://shtools.github.io/SHTOOLS/).

---

## Acknowledgments, licence, changelog

Parts of this toolkit, including the eigenfunction solver methodology and the reference soliton profile,
were inspired by Mathematica code generously provided by [J. Luna Zagorac](https://orcid.org/0000-0003-4504-1677).
The theoretical foundation of the eigenmode decomposition is Zagorac, Sands, Padmanabhan & Easther, Phys. Rev.
D 105, 103506 (2022).

The unit conversions (`units`, `pyul_run`) are adapted from PyUltraLight / PyUltraLight 2, and the bundled
soliton profile is PyUltraLight's standard profile; [NOTICE](NOTICE) records the origin of these parts and
credits their authors.

Licensed under the MIT licence; see [LICENSE](LICENSE) and [NOTICE](NOTICE). All changes are listed in
[CHANGELOG.md](CHANGELOG.md).
Issues and pull requests are welcome on GitHub.
