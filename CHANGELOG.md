# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

Every entry has an ID and a fix policy:

- **A** (fix outright): crashes, portability or import failures, wrong results in code paths the paper did
  not use, unit bugs in helpers, unsafe behaviour, missing metadata.
- **B** (keep the legacy default, add an opt-in switch): anything that would change numbers already
  published or used for the paper.
- **C** (fix by default, keep a legacy switch): not used in this release.

Default numerics are unchanged. With default options, the a_lm and c_nlm values of the historical chain
(`process_file` -> JSON -> `create_sh_hdf5` -> `load_a_lm` -> `compute_cnlm` -> `save_cnlm_to_h5`) are
bitwise identical to version 0.2.0 when run in the same process. Across processes, pyshtools' FFT
already differed by about 1e-16 relative before this release (see WP2-19).

## [0.3.0] - 2026-10-18

### Added

- **WP2-19 (new): `Eig_ULDM_packages/alm_kernel.py`**: a one-pass a_lm(r) kernel for one snapshot cube,
  `compute_alm_cube(psi_cube, dx, r_values, l_max, num_points=300, angular_grid='legacy', center='com',
  workers=None)` (and the reusable `AlmKernel`).
  - It computes the COM once and the cubic spline prefilter once (`scipy.ndimage.spline_filter`).
  - It samples shells in a thread pool (`map_coordinates(prefilter=False)` releases the GIL) and calls
    `SHExpandDHC` exactly as `functions3.process_file` does.
  - The result is bitwise identical to `process_file` within one process, verified on SciPy 1.13.1 and
    1.18.1 for 1 to 64 threads and any point chunking.
  - pyshtools' `SHExpandDHC` is deterministic within a process but not across processes (last-bit
    differences, about 1e-17 absolute; consistent with measured FFT planning). This was already true
    for the historical multi-process chain.
- **WP2-20 (new): `Eig_ULDM_packages/cnlm_snapshot.py`**: per-snapshot building blocks for in-simulation
  or streaming use.
  - `CnlmProjector` / `cnlm_from_alm_snapshot` compute c_nlm of one snapshot exactly as `compute_cnlm`
    does for that time index.
  - `alm_counts_roundtrip` reproduces the a_lm HDF5 storage round trip. `ylm = a*counts` then
    `ylm/counts` is **not** an identity: about 14 % of values change in the last bit.
  - `legacy_alm_radial_grid` gives the code-unit shell radii of `compute_c_nlm.ipynb`, with an r_max check.
  - `write_cnlm_h5` writes the `save_cnlm_to_h5` layout and `write_alm_h5` the `create_sh_hdf5` layout.
    Both write to a temporary file, rename it, and refuse to overwrite without `overwrite=True`.
- **WP2-21 (new): local pipeline `python -m Eig_ULDM_packages.pipeline RUN_DIR`** (console script
  `uldm-cnlm`), with the helper module `Eig_ULDM_packages/pyul_run.py`.
  - It reads `config.uldm` / `reproducibility.uldm`: time axis, axion mass, soliton mass, and the
    resampled-box geometry when PyUL_SK >= 30.50 records it.
  - It can build the eigen file with the paper defaults (n_grid 5120 including r=0, lmax 10, n_eig 25,
    r_max = half the resampled box).
  - It computes a_lm with WP2-19 and writes one checkpoint per snapshot, so it is resumable.
  - It writes `c_nlm.h5` and, optionally, the a_lm file, each with provenance attributes.
  - Final outputs are never overwritten without `--force`.
  - Templates are in `examples/pbs/run_cnlm_pipeline.pbs` and `examples/slurm/run_cnlm_pipeline.slurm`
    (`set -euo pipefail`, no host pinning, unique logs).
- **WP2-25 (A): `tests/`**: a pytest suite (15 test modules plus `conftest.py` and `_legacy_chain.py`) covering every item below, including bitwise
  equivalence of the new pipeline with the historical chain on synthetic data and on a real PyUL run
  (`ULDM_EQUIV_RUN_DIR`).
- `Eig_ULDM_packages/_compat.py` (`trapezoid`, `sph_harm_ylm`, `mp_context`, `default_workers`,
  `prefix_folder`), `Eig_ULDM_packages/_h5meta.py` (root-attribute provenance helpers), and
  `Eig_ULDM_packages.__version__`.
- **WP3-01 (new): `Eig_ULDM_packages/cnlm_inline.py`**, the in-simulation recorder behind PyUL_SK
  30.50 `evolve(Cnlm=True)`: `CnlmInlineSettings.validate` (dependencies, parameters, eigen file or
  unambiguous automatic build, output paths; before PyUL touches any output) and
  `InlineCnlmRecorder.setup / record / finalize` (axion mass and RS geometry from the run record,
  paper shells, eigen file built with the paper defaults when not given, r_max and lmax checks,
  per-save checkpoint in the pipeline format, `c_nlm.h5` and optional a_lm file written with
  `write_cnlm_h5` / `write_alm_h5`; root attribute `source='pyul-inline'` plus PyUL version, core
  sha256, eigen-file sha256, kernel parameters and RS geometry). Bitwise equal to the pipeline on
  the same cubes (tests/test_cnlm_inline.py here, tests/test_cnlm_inline.py in PyUL_SK). (A)
- **WP3-02: pipeline `--from-inline`** takes every parameter from `Outputs/Cnlm/cnlm_inline.json`
  and assembles c_nlm from the inline checkpoints (`source_id='pyul-inline'`, accepted without a
  3WfnRS file), computing only snapshots without a checkpoint from their file; usable after a crash.
  `--r0`, `--num-points`, `--angular-grid`, `--cnlm-out` defaults are now resolved inside
  `run_pipeline` (same values). (A)
- **WP3-03: `cnlm_snapshot.kernel_fingerprint`, `checkpoint_path`, `write_alm_checkpoint`,
  `read_alm_checkpoint`, `INLINE_SOURCE_ID`**, shared by the pipeline and the recorder (same
  fingerprint dictionary and hash as before). (A)
- **WP3-04: `AlmKernel.compute(..., shell_block=None)`** and `sample_shells(..., shells=None)`:
  sample and expand the shells in blocks (bounded memory: about 0.47 GB instead of about 1.2 GB
  extra at the paper's 257 shells x 300 x 600 points); bitwise identical values (tested). (A)

- **R9-14 (new): `Eig_ULDM_packages/trajectories.py`**, the public black-hole trajectory loader. `load_run`
  reads a raw PyUL run directory (NBody/ULDCOM/DF frames in npy, npz or hdf5, energies, explicit snapshot
  times) or a trajectory bundle and returns one `PyULRun` object; `export_bundle` / `load_bundle` write and
  read a single-file `.npz` bundle per run (no pickled objects, verbatim `config.uldm` and
  `reproducibility.uldm`); `distance_from_origin_pc`, `distance_from_uldm_com_pc` and
  `verify_length_conversion`; `legacy_time_axis=True` reproduces the old notebooks' `k*duration/(save_number+1)`
  axis. Command line: `python -m Eig_ULDM_packages.trajectories info|export`. Run reading, units and the
  axion-mass policy come from `pyul_run.py` (WP3-05).
- **R9-15 (new): `paper_figures/`**: one script per printed figure of the Stone Skipping paper
  (`fig01_*.py` ... `fig12_*.py`), the shared helpers `_common.py` and `_seeded_binary.py`,
  `make_all_figures.py` (all figures, one process each), `export_bundles.py` (raw run directories to
  trajectory bundles), `figures_config.json` (layout of the data deposit), `figures_config.raw_example.json`
  (raw run directories) and `README.md` (conventions, data layout, validation against the published figures).
  Each script writes a `<basename>.provenance.json`.
- **R9-16 (new): repository metadata**: `.zenodo.json` (software deposit metadata), `.gitattributes` (LF
  line endings for text files, binary data files marked binary) and `MANIFEST.in` (R9-12).

### Changed

- **WP3-05 (A, F15): one run loader and one axion-mass policy.** `Eig_ULDM_packages/pyul_run.py` is
  the single implementation of run-directory reading (`find_run_dir`: run folder, its `Outputs`, or a
  save_path with exactly one run; `read_run_metadata`; `PyULRun` now also keeps the JSON texts),
  PyUL's unit system (`code_units`, `convert_to_code`, ...), the time axis (`pyul_time_axis_myr`,
  `find_snapshot_times`) and the axion mass (`axion_mass_from_reproducibility`,
  `resolve_axion_mass`). `trajectories.py` re-exports these names unchanged and keeps only the frame
  and bundle code; `pipeline.py`, `cnlm_inline.py` and `compute_coeff.py` use the same functions.
  Policy everywhere: the mass recorded in `reproducibility.uldm` first; then an explicit argument
  (`axion_mass_ev=` / `--axion-mass-ev`), which must agree with a record (ValueError otherwise);
  `PYUL_AXION_MASS_EV` only as the last resort, with a RuntimeWarning (also a warning when it
  disagrees with the record). Behaviour changes: the pipeline no longer lets `--axion-mass-ev`
  override a recorded mass; `trajectories.load_run` now falls back to the environment variable
  (with a warning) instead of raising when neither a record nor an argument exists.
- **WP3-06 (A): pipeline time axis** = PyUL's real save times `Outputs/snapshot_times_Myr.npy` when
  the file exists (PyUL_SK >= 30.50) and covers the snapshots, else
  `linspace(0, duration, save_number+1)[snapshots]` as before; recorded in the `time_axis` attribute.
  `PyULRun.snapshot_times_myr(indices, return_source=False)`.
- **WP2-18 (B, T17): angular sampling before `SHExpandDHC`**.
  - The historical grid `theta = linspace(0, pi, N)`, `phi = linspace(0, 2pi, 2N)` (both endpoints) is
    not the Driscoll-Healy grid. It stays the default (`angular_grid='legacy'`) so published numbers
    reproduce.
  - `angular_grid='dh'` (`theta_i = pi i/N`, `phi_j = pi j/N`) is available in
    `functions3.compute_sh_coeffs` / `process_file`, the kernel, the pipeline and `compute_coeff.py`.
  - Measured at N = 300 on pure Y_lm: Y00 exact. For l >= 1, the amplitude error is 0.1-0.4 %, the
    phase offset is m*pi/(2N-1), and leakage is 0.3-4 %. 'dh' is exact to 1e-15.
- **WP2-17 (B, T16): eigen solver outer boundary**. `solve_h` / `make_all_eigs` gain
  `outer_boundary='legacy'` (default: the last grid point is an unknown, so the Dirichlet wall is
  effectively at r_max + h) or `'rmax'` (u(r_max) = 0 exactly; converges at second order to the
  fixed-domain spectrum).
- **WP2-27 (B): `load_soliton_file(path, mass0=None)`** can override the bundled profile's rounded
  `mass0 = 3.883`; the exact value is 3.8827652755822006, and the rounded one makes the scaled soliton
  about 6e-5 too light. The pipeline exposes it as `--soliton-mass0`. The default keeps the file value
  (published eigenfunctions), and `soliton.h5` is unchanged.
- **WP2-07 (A, T7): failures are reported**.
  - `functions3.process_file(..., raise_on_error=True)` now re-raises after logging. `False` restores
    the old print-and-continue behaviour.
  - `alm_utils.create_sh_hdf5` raises `RuntimeError` on missing or unreadable JSON files and leaves no
    output. `allow_missing=True` restores zero-filling and records `missing_count`, `missing_per_time`
    and `missing_samples` as root attributes. The file is written to `.tmp` and renamed.
- **WP2-09 (A, T9): `load_a_lm` time axis**.
  - New keyword-only `times`, `snapshot_indices`, `save_interval` and `time_source` ('auto', 'legacy'
    or 'file').
  - Files with a `times` attribute (toolkit >= 0.3.0) use their own times. Historical files keep
    `linspace(0, final_time, n)`.
  - The r grid is also taken from the `r_values_code` attribute when present.
  - The returned dict gains `snapshot_indices`, `time_source` and `attrs`.
- **WP2-10 (A, T10): `_parse_rmax_from_filename`** only accepts an explicit `rmax` token (`rmax0.2`,
  `rmax_0.2`, `r_max=0.2`). The old rule read the paper's `PyUL_psi_alm_0.02_mass170_256.h5` as
  r_max = 0.02; such names now return `None` and `load_a_lm` asks for `r_al`.
- **WP2-14 (A, T14)**:
  - `resample_cnlm_time(method='nearest')` is now pure nearest-neighbour. It silently applied
    Savitzky-Golay smoothing before; the paper notebooks use 'nearest-savgol' explicitly and are
    unaffected.
  - The `save_cnlm_to_h5` docstring now gives the key order (n, l, m). Metadata dicts are stored as
    JSON strings.
  - `compute_cnlm(num_workers>1)` sends one (l, m) a_lm matrix per task instead of pickling the whole
    a_lm into every worker, and reuses each radial interpolation for all n. The results are bitwise
    identical.
- **WP2-12 (A, T12): `compute_coeff.py`** is configurable with argparse, `ULDM_*` environment variables
  or the constants at its top.
  - Paper defaults: snapshots 0..save_number (from `../config.uldm`), l_max 10, num_points 300.
  - The worker count comes from the environment or CPU affinity (default CPUs // 4) instead of the
    hard-coded 56.
  - The 3WfnRS geometry is read from `reproducibility.uldm` (PyUL_SK >= 30.50), else it must be given
    explicitly.
  - It runs one process pool for all tasks behind a `__main__` guard, reports failures with a non-zero
    exit code and a summary JSON, and offers `--skip-existing`. The `r_0_values = np.load(...)` template
    line used by Main.ipynb is kept.
- **WP2-13 (A, T13): `units`**.
  - `convert_back(v, 'c', 'v')` is fixed (it was inverted: a round trip of 0.001c gave 4e-8).
  - One `UserWarning` is emitted at the first conversion when `PYUL_AXION_MASS_EV` is unset (default
    1e-22 eV; the paper used 1e-21 eV).
  - New `set_axion_mass()`, `get_axion_mass_ev()` and `axion_mass_source()`. `set_axion_mass(1e-21)` is
    bitwise equal to importing with the environment variable set.
- **WP2-15 (A, T15): `cnlm_spectral_analysis`**.
  - The `plot_cnlm_multimode` legend stated `m=0>m_+>m_-` while drawing m_- wider than m_+. The width
    order is now derived from the widths actually drawn, which are parameters `lw_zero`, `lw_pos` and
    `lw_neg` with unchanged defaults.
  - New `time_unit` parameter (default 'Myr') in `export_prd_figure`, `plot_cnlm_prd_single` and
    `plot_cnlm_multimode`.
  - `plot_cnlm_prd_single(xlim=None)` uses the full time range (the default is still (0, 3000)).
  - A shadowed `m` was renamed. Default figures are unchanged.
- **WP2-16 (A, T16): `uldm_eig1`**.
  - The `solve_h` docstring now gives the real default tol (1e-16; `make_all_eigs` passes 1e-10).
  - `compute_potential_from_grid` warns when it infers `dx = 2 r_max/N` and gains `cell_centered=True`
    (pass False for PyUL's node grid).
  - `save_eigs_to_h5(..., metadata=None)` writes provenance root attributes (the `/ell` layout is
    unchanged).
  - The quadrature mix in `compute_potential` is documented.
- **WP2-08 (A, T8): provenance metadata** as root attributes only: r grid and units, times, snapshot
  indices, kernel parameters, RS geometry, axion mass, eigen-file SHA-256, and software versions. No
  new groups or datasets; dataset layouts are unchanged. Arrays longer than 4096 elements are split
  into numbered attribute chunks to stay below the HDF5 limit.
- **WP2-23 (A, P1): packaging**.
  - Version 0.3.0, SPDX `license = "MIT"` with `license-files` (builds with setuptools >= 77, which
    supports Python 3.9).
  - Extras: `anim` (moviepy) and `dev` (pytest, nbformat, jupyter, moviepy).
  - Console script `uldm-cnlm` and pytest configuration.
- **WP2-24 (A): notebooks**. Template bugs and wrong comments are fixed, and all stored outputs (which
  contained private server paths) are stripped. Nothing else was rewritten.
  - `Main.ipynb`: snapshot range 0..save_number from `config.uldm` (was 500..1000). Step 3 reuses Step
    1's `r_0_values.npy` instead of rebuilding the grid with a hard-coded r_max. "resolution" and
    "box_length" refer to the resampled 3WfnRS cube. `create_sh_hdf5` records times and units.
  - `compute_c_nlm.ipynb`: the axion mass is set before the first toolkit import. The checked
    `rescale_r_to_eigen_rmax` replaces the silent r_al rescale. The final_time, dense-resampling,
    line-width and cell-reference text is corrected.
  - `f_nl_python.ipynb`: the axion mass is set before imports. Corrected: the grid comment (it includes
    r=0, spacing rmax/(n_grid-1)), the file-name references, and the HDF5 layout description.
- **WP2-22 (A): `submit_all_jobs_example.sh`**.
  - `set -euo pipefail`; inputs are checked before submission and `qsub` errors stop the script.
  - Generated `compute_coeff<i>.py` and `.pbs` files are no longer deleted (they are kept in
    `pbs_jobs_<stamp>/`), and logs are time-stamped.
  - Host pinning is opt-in (empty `NODE_LIST`), `Main.ipynb` replaces the obsolete `a_lm.ipynb`
    reference, and `COEFF_ARGS` passes the RS geometry.

### Fixed

- **WP2-01 (A, T1)**: `np.trapz` (removed in NumPy 2) is replaced by `_compat.trapezoid` (same formula)
  in `uldm_eig1.compute_potential`, `c_nlm_integrator.compute_cnlm` and its worker, and
  `wavefunction_tools.compute_mode_fractions`.
- **WP2-02 (A, T2)**: the SciPy Y_lm fallback in `wavefunction_tools` uses `sph_harm_y(l, m, theta, phi)`
  (SciPy >= 1.15) or the removed `sph_harm(m, l, phi, theta)`. Note the argument order. Verified
  against pyshtools.
- **WP2-03 (A, T3)**: `wavefunction_tools` no longer requests `get_context('fork')`, which does not
  exist on Windows and is unsafe on macOS. `_compat.mp_context()` uses 'fork' on Linux (the historical
  behaviour) and 'spawn' elsewhere, with override `ULDM_MP_START_METHOD`. The workers were already
  picklable.
- **WP2-04 (A, T4)**: `functions3` imports `matplotlib` and `moviepy` lazily, inside the plotting and
  video functions, and imports `cryptography` only if it is installed (it is not a dependency).
  `compute_coeff.py` now imports on a minimal install. `from functions3 import *` no longer exports
  `plt`, `ImageSequenceClip`, `FormatStrFormatter` or `FixedLocator`.
- **WP2-05 (A, T5)**: invalid escape sequences in docstrings (`\e`, `\h`) are now raw strings
  (`c_nlm_integrator` x5, `uldm_eig1` x1), which removes SyntaxWarnings on Python >= 3.12.
- **WP2-06 (A, T6)**: `file_prefix.split('/')` is replaced by `_compat.prefix_folder`, which handles
  `\` and absolute prefixes. The result is identical for the historical relative prefixes.
- **WP2-11 (A, T11)**: new `c_nlm_integrator.check_rmax_consistency` and `rescale_r_to_eigen_rmax`. The
  latter refuses to stretch the a_lm radii onto the eigen grid when the two outer radii differ by more
  than 1e-6 relative.
- **WP2-26 (A)**: `wavefunction_tools` raises an English error message (it was Chinese text, unreadable
  on some consoles), and `.gitignore` covers the files the pipeline generates.

### Review fixes (R9, 2026-10-18)

Found by the adversarial review of 0.3.0 before release. Default numerics are unchanged: the historical chain,
the pipeline and the inline recorder give the same values as before for every complex128 (PyUL) cube.

- **R9-01 (A): `make_all_figures.py` with relative paths.** The figure scripts ran with the working directory
  `paper_figures/`, but `--data-root`, `--out-dir` and `--config` were passed on unresolved. The documented
  command `make_all_figures.py --data-root . --out-dir figures_out` therefore failed for 11 of 12 figures, and
  figures went into `paper_figures/<out-dir>` while the status file went to `./<out-dir>`. The three paths
  are now resolved against the caller's working directory, and the scripts run in that directory.
- **R9-02 (A): `compute_coeff.py --skip-existing` checks the kernel parameters.** The JSON files record only
  `source_file`, `r_0` and `coeffs`, so a rerun with another dx, `num_points` or angular grid skipped every
  shell and mixed incompatible a_lm silently. Each JSON tree now has a sidecar
  `alm_coefficients/<folder>/kernel_params_<prefix>lmax_<l>.json` (dx, RS resolution, `num_points`,
  angular grid, `l_max`). A rerun with other values into the same tree is refused (also without
  `--skip-existing`, which would mix the tree). `--skip-existing` on a tree made before this check (no
  sidecar) is refused too; `--no-params-check` restores the old unchecked behaviour and rewrites the sidecar.
  The summary JSON records `skipped_existing`. `functions3.save_sh_coeffs` writes each JSON to a temporary
  file and renames it, so a killed job no longer leaves a truncated file (same content).
- **R9-03 (A): 0.2.0 JSON trees with an absolute prefix stay readable.** 0.2.0 put the JSON files of an
  absolute prefix under `alm_coefficients/<file_prefix.split('/')[0]>/` (`alm_coefficients/r0_<r>/...` on
  POSIX); WP2-06 moved new files to `alm_coefficients/<snapshot folder>/`. `create_sh_hdf5` and
  `load_alm_from_file` now fall back to the 0.2.0 location when the new file does not exist
  (`_compat.alm_json_path`). Relative prefixes are unchanged.
- **R9-04 (A): one-pass kernel on single-precision cubes.** `process_file` samples a complex64 cube into
  float32 (`map_coordinates` returns the input dtype), the kernel sampled in float64, so the "bitwise equal to
  `process_file`" claim did not hold for complex64 (about 5e-9 relative). `AlmKernel` now returns the samples
  in the cube's real dtype as well; it is bitwise equal to `process_file` for complex128, complex64, float64
  and float32 cubes. Values for complex128 cubes (everything PyUL writes) are unchanged.
- **R9-05 (A): inline c_nlm, eigen file inside `Outputs`.** `evolve` deletes and recreates `<run>/Outputs`
  before the run starts, so a `CnlmEigenFile` stored there (the previous run's automatic `Outputs/Cnlm/eigs_*.h5`,
  or `Outputs/cnlm_pipeline/eigs_*.h5` from `uldm-cnlm --build-eigen`) was deleted and the run died after the
  wipe. `CnlmInlineSettings.validate` now refuses such a file with a clear message before anything is deleted
  (new keyword `outputs_replaced=True`).
- **R9-06 (A): inline recorder memory.** The c_nlm of each save is kept as one complex128 vector (about 48 KB
  at l_max 10, n_eig 25) instead of a dict of NumPy scalars (about 450 KB); `c_nlm.h5` is unchanged (bitwise).
- **R9-07 (A): checkpoints across toolkit versions.** The a_lm checkpoint fingerprint hashed the package
  version, so a toolkit update made the checkpoints of a crashed inline run unusable. It now hashes
  `cnlm_snapshot.KERNEL_FORMAT_VERSION` = "0.3.0" under the same key, which changes only when the kernel or
  the checkpoint format changes; 0.3.0 checkpoints keep their hash. The package version is recorded in
  `pipeline_params.json` for information only. `uldm-cnlm` now replaces an existing `pipeline_params.json`
  (with `--force`) only after every checkpoint was validated or computed, so a failed `--force` call no longer
  rewrites the record of good checkpoints.
- **R9-08 (A): `--from-inline` checks the eigen file.** The eigen file named in `cnlm_inline.json` is
  compared with the SHA-256 recorded by the run; a rebuilt or replaced file is refused (`--eigen-file` to
  choose a file explicitly, `--force` to continue with a warning).
- **R9-09 (A): runs saved with `Save Options Number = -1`.** PyUL replaces -1 (save every step) by the step
  count in memory only, so `uldm-cnlm` and `--from-inline` saw save number -1 (no checkpoint accepted,
  `IndexError` for a plain run). New `PyULRun.effective_save_number()` counts the saves from
  `Outputs/snapshot_times_Myr.npy`, the snapshot files and the checkpoints; the time axis of such runs comes
  from `snapshot_times_Myr.npy` (a clear error if it is missing). `parse_snapshot_spec` raises on a negative
  save number instead of returning an empty list. The snapshot times are now resolved before the a_lm loop.
- **R9-10 (A): one multiprocessing policy.** `compute_cnlm`, `create_sh_hdf5` and `create_animation` used
  the interpreter's default start method and ignored `ULDM_MP_START_METHOD`; they now use
  `_compat.mp_context()` like `wavefunction_tools` (fork on Linux, spawn elsewhere). Results are unchanged.
- **R9-11 (A): console output on cp1252.** `print_spectral_summary` printed non-ASCII symbols and failed with
  `UnicodeEncodeError` when stdout was redirected on a cp1252 Windows locale. Its messages are ASCII now and
  are printed with a fallback that replaces unencodable characters (`SpectralAnalysisResult.fft_note` is
  unchanged); the HDF5 tree printer of `alm_utils` uses ASCII as well.
- **R9-12 (A): sdist.** New `MANIFEST.in`: the sdist now contains `tests/conftest.py`,
  `tests/_legacy_chain.py` and the files the tests read (`paper_figures/`, `examples/`, notebooks,
  `compute_coeff.py`, `submit_all_jobs_example.sh`), so `python -m pytest` runs from an unpacked sdist.
- **R9-13 (A): documentation.** The `--axion-mass-ev` help of the figure scripts said the environment is never
  read; `PYUL_AXION_MASS_EV` is the documented last resort (with a RuntimeWarning). `figures_config.json` and
  `paper_figures/README.md` state that `res256` and `single256_30000` are the same simulation
  (`20260119_062714@256`). The README says that the three c_nlm routes agree for the same kernel parameters
  (the default shell counts agree for 256^3 cubes), and that the automatically built eigen file matches the
  paper's in its datasets, not in its file hash (0.3.0 adds provenance attributes).

## [0.2.0]

Version used for the Stone Skipping paper analysis (GitHub main @ fea20e7).
