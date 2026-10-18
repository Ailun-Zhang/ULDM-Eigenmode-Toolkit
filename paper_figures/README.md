# Paper figures: *Stone Skipping Black Holes in Ultralight Dark Matter Solitons*

These scripts regenerate every printed figure of the paper ([arXiv:2602.11512](https://arxiv.org/abs/2602.11512);
13 figures, Figs. 1-13) from the small deposited data products: per-save black-hole trajectories, the
`c_nlm(t)` coefficient files and the radial eigenbasis.
The large 3-D wavefunction outputs are not needed.

Each script reproduces the content and style of the code that made the published figure, and its default
options are the ones that figure used. Where the original did something unusual (a compressed time axis,
a smoothed and resampled coefficient series), the script does the same by default, exposes it as an
option, and documents it below. None of the scripts adds smoothing, phase alignment, offsets or rescaling
that the original figure did not have.

Script names and the `figures_config.json` keys use the **printed figure numbers** of the paper. There are 12
scripts for 13 figures, because `fig06_fig07_cnlm_single_modes.py` makes the two single-mode figures 6 and 7
(key `fig06_fig07`). LaTeX labels of the manuscript source are quoted only as labels; they are not figure
numbers. Earlier working documents (before 2026-10-18) counted Figs. 6 and 7 as one figure, so their
"Fig. 7" ... "Fig. 12" are printed Figs. 8 ... 13.

| Printed figure | Script (key) | Output file(s) |
|---|---|---|
| Fig. 1 | `fig01_mass_sequence_radius.py` (`fig01`) | `distance_from_origin_vs_time_v7_256` |
| Fig. 2 | `fig02_resolution_robustness.py` (`fig02`) | `resolution_robustness_round2_192` |
| Fig. 3 | `fig03_eigenfunction_grid.py` (`fig03`) | `f_nl_grid_3x5` |
| Fig. 4 | `fig04_equal_mass_binary.py` (`fig04`) | `0.2_Binary_pc_v2` |
| Fig. 5 | `fig05_cnlm_3x3.py` (`fig05`) | `c_nlm_3x3_n012_v7` |
| Figs. 6 and 7 | `fig06_fig07_cnlm_single_modes.py` (`fig06_fig07`) | `cnlm_n1_l1_mneg1_original_v3` (Fig. 6), `cnlm_n1_l0_m0_original_v3` (Fig. 7) |
| Fig. 8 (a-d) | `fig08_reconstructed_background.py` (`fig08`) | `stone_skipping_{compare_n0,compare_n1,compare_n2,collection2}_pc_v2` |
| Fig. 9 | `fig09_higher_mode_test.py` (`fig09`) | `distance_3d_1x2_pc_v3` |
| Fig. 10 | `fig10_reduced_seeded_binary.py` (`fig10`) | `equal_mass_reduced_seeded_vs_unseeded` |
| Fig. 11 | `fig11_fullsp_seeded_binary.py` (`fig11`) | `equal_mass_seeded_vs_unseeded_fullSP` |
| Fig. 12 | `fig12_cnlm_time_spectrum.py` (`fig12`) | `cnlm_time_spectrum_n1_ell1_m-1_v3` |
| Fig. 13 | `fig13_resonance_roots.py` (`fig13`) | `f_x` |

The output files keep the file names of the published figures. Known issues of the code versions behind the
figures (for example the resampled-box radius error that affects the c_nlm of Figs. 2 and 12) are explained
in [docs/conventions_and_errata.md](../docs/conventions_and_errata.md) of the toolkit (online:
https://github.com/Ailun-Zhang/ULDM-Eigenmode-Toolkit/blob/v0.3.0/docs/conventions_and_errata.md; items E1-E6,
cited as "errata E<n>" below); the scripts reproduce the published figures as they are.

## Quick start

```bash
pip install numpy scipy matplotlib h5py          # Python >= 3.9; tested with 3.9.25/mpl 3.9.3 and 3.13/mpl 3.11
pip install -e .                                 # this toolkit (eig-uldm-packages 0.3.0), from the repository root
python paper_figures/make_all_figures.py --data-root /path/to/StoneSkipping_PRD_data_v1 --out-dir figures_out
python paper_figures/fig08_reconstructed_background.py --data-root /path/to/StoneSkipping_PRD_data_v1 --formats pdf --panels compare_n1
python paper_figures/fig04_equal_mass_binary.py --data-root /path/to/StoneSkipping_PRD_data_v1 --legacy-time-axis false   # PyUL time axis
```

`/path/to/StoneSkipping_PRD_data_v1` is the unpacked data deposit *Data for "Stone Skipping Black Holes in
Ultralight Dark Matter Solitons"* (Zenodo, version 1, [doi:10.5281/zenodo.23454343](https://doi.org/10.5281/zenodo.23454343)), which also contains a copy of these scripts.

Every script accepts these options:

| option | meaning |
|---|---|
| `--data-root` | root of the data tree. All paths in the config are relative to it. |
| `--config` | figure/run configuration (default `paper_figures/figures_config.json`, the deposit layout) |
| `--out-dir` | output directory (default `./paper_figures_output`) |
| `--formats` | comma-separated list from `pdf,eps,png,svg` (default `pdf,eps,png`) |
| `--basename` | output file stem (default: the published file name) |
| `--png-dpi` | PNG resolution (default: the original script's value, else 300) |
| `--prefer raw\|bundle` | read raw run directories or single-file trajectory bundles when the config lists both |
| `--axion-mass-ev` | fallback axion mass for runs whose `reproducibility.uldm` does not record it |
| `--no-provenance` | do not write `<basename>.provenance.json` |

Figure-specific options are listed by `--help`. Each run also writes `<basename>.provenance.json`, which
records the input files (relative path, run key, original run ID), the SHA-256 of every c_nlm/eigenbasis
file, the PyUL version, axion mass and time-axis convention of every run, the options used, and the
software versions.

## Data layout

### Deposit layout (`figures_config.json`, the default)

```
<data-root>/runs/<run key>.npz                 one trajectory bundle per run (see below)
<data-root>/cnlm/<run key>_c_nlm.h5            c_nlm(t) files, byte-identical copies of the files the runs wrote
<data-root>/eigen/eigs_mass_170_Eig_25_lmax_10_nig_5120_py2__<sha8>.h5   the two radial eigenbasis files
<data-root>/derived/...                        published-figure inputs, used for validation and fig02 --source csv
<data-root>/provenance/runs.csv                per-run parameters, PyUL versions and original run IDs
```

`figures_config.json` maps every run key to its `bundle` and records the original PyUL run folder name in
`original_run_id` (for example `20251016_185657@256`); `printed_figures: false` marks the three
supplementary runs that no default figure plots. It also lists the HDF5 files with their original file
name and a SHA-256 prefix, which is checked when the file is read. To use a different layout, edit only the
config (or pass your own with `--config`).

### Raw PyUL run directories (`figures_config.raw_example.json`)

The same runs can be read from the run folders PyUL writes. `figures_config.raw_example.json` uses the
placeholder layout

```
<data-root>/raw/<run key>/<original run ID>/
    config.uldm, reproducibility.uldm
    Outputs/NBody/NTM_#000..NNN.npy, Outputs/ULDCOM/UCM_#..., Outputs/DF/DYF_#...
    Outputs/*list.npy (energies), Outputs/ULDMass.npy
    Outputs/c_nlm*.h5, Outputs/eigs_mass_170_Eig_25_lmax_10_nig_5120_py2.h5   (where used)
<data-root>/runs/<run key>.npz        optional bundles (used when a raw directory is absent)
<data-root>/derived/...
```

Move or link the run folders into this layout, or edit the `path` entries, and pass
`--config paper_figures/figures_config.raw_example.json`. Figures made from raw directories and from
bundles are identical (checked for every figure, see Validation).

### Trajectory bundles

```bash
python paper_figures/export_bundles.py --config paper_figures/figures_config.raw_example.json \
    --data-root RAW_ROOT --out-root DEPOSIT_ROOT --verify
```

writes one compressed `.npz` per run with a raw `path` (about 0.07 MB each; 1.8 MB for all 26 runs). A
bundle holds the frame indices, the NBody/ULDCOM/DF state vectors in code units, the energy arrays, ULDM
mass, particle masses, any explicit snapshot times, the verbatim `config.uldm` and `reproducibility.uldm`,
and the run key and original run ID. It contains no pickled objects. A deposit can therefore ship one file
per run instead of about 1500. When a raw run directory is absent, the scripts read the bundle
automatically. Bundles are read and written with `Eig_ULDM_packages.trajectories` (`load_run`,
`export_bundle`, `load_bundle`; CLI `python -m Eig_ULDM_packages.trajectories info|export`).

## Conventions (Eig_ULDM_packages/trajectories.py)

* **Time axis.** PyUL saves frame `k` at `t_k = start + k*duration/save_number`, and `load_run` uses this
  by default. PyUL >= 30.50 also writes `Outputs/snapshot_times_{Myr,code}.npy`, which is used when
  present. The old analysis notebooks (`plot.ipynb`, `Test_PyUL_alan_2.ipynb`,
  `Test_PyUL_alan_Binary.ipynb`) used `t_k = k*duration/(save_number+1)`. For 500 saves this compresses
  the axis by 500/501, so the last frame is drawn at 2994.01 Myr instead of 3000 Myr.
  `legacy_time_axis=True` (`--legacy-time-axis`) reproduces that convention. The published Figs. 4, 8 and
  9 used it. So did the source PDFs that Figs. 10 and 11 were digitised from. This was verified from the
  vector paths of the published PDFs (see Validation and errata E5).
* **Units.** Code length is converted to pc with the axion mass recorded in each run's
  `reproducibility.uldm` (`"Axion Mass"` in kg and `"m22"`, which must agree). If neither is recorded,
  `--axion-mass-ev` must be given. The `PYUL_AXION_MASS_EV` environment variable is used only as a last
  resort, with a `RuntimeWarning`; every paper run records its mass, so the figures never use it. At 1e-21
  eV, 1 code length = 12133.0958 pc (`verify_length_conversion()`).
* **Observables.** `distance_from_origin_pc(body)` is |x_BH| from the simulation origin, which is the
  initial total centre of mass. `distance_from_uldm_com_pc(body)` is |x_BH - x_ULDCOM|.
* **c_nlm normalisation.** `|c|^2 / M_code`, where M_code is the soliton mass of 12e6 Msun in code units
  (170.3149822613627 at 1e-21 eV). The axion mass comes from the run that owns the c_nlm file.

## Figures

"Published" means the file included by the manuscript source (`main_new2.tex`). All trajectory runs are
3000 Myr with 500 saves unless noted otherwise. Notebook and script names refer to the original analysis
code; they are cited for provenance and are not part of this repository unless stated.

### Fig. 1 (label `fig:1`): `fig01_mass_sequence_radius.py` -> `distance_from_origin_vs_time_v7_256`
* **Runs:** `fig1_m1`, `fig1_m1p5`, `single256_3000` (2 %), `fig1_m3`, `fig1_m4`, `fig1_m5`. All are
  N=256 with a 5 kpc box.
* **Observable:** |x_BH| from the origin, in pc, from all 501 samples over 0-3000 Myr.
* **Time axis:** the PyUL axis, as published.
* **Style:** the original figure script `refine_figure1.py` of the 2026-10-08 figure update: Arial,
  9.8 x 6.1 in, six colours and dash patterns, no legend, PNG at 210 dpi.
* **Differences:** none in the data. The 2 % run is PyUL 26.32 started from a precompiled deboosted IC;
  the other five are PyUL 30.4 started from an analytic soliton (recorded, not changed).

### Fig. 2 (label `fig:resolution_robustness`): `fig02_resolution_robustness.py` -> `resolution_robustness_round2_192`
* **Runs:** `res128` (the N=128 rerun, run ID `20260920_131633@128`), `res192` and `res256`, with their
  `c_nlm` files. The window is 0-3000 Myr at 60 Myr cadence, 51 samples each.
* **Observables:** (a) r_sys = |x_BH| from the origin, in pc. (b) P_dip = sum over n<=2 and m of
  |c_n1m|^2 / M_code.
* **Style:** the original figure script `make_resolution_figure.py` of the 2026-10-08 figure update.
  `--source csv` replots `derived/resolution_128_192_256.csv`, which was the published input.
* **Differences:** none in the data. The published N=128 curves come from the rerun `res128` (PyUL 30.4).
  `--run-128 res128_jan2026` uses the earlier January-2026 N=128 run (`20260118_004908@128`, PyUL 26.32,
  supplementary) instead. Its r_sys differs by up to 0.053 pc and its P_dip by up to 3.5 %.
* **Known issue:** the three c_nlm files come from PyUL 30.40 (N=128, 192) and 30.38 (N=256), whose
  resampled box makes the a_lm radii 0.79/0.52/0.39 % too large. Recomputed with the true spacing, P_dip is
  0.6-2.4 % lower; the figure (and this script) shows the published values. See errata E4.

### Fig. 3 (label `fig:2`): `fig03_eigenfunction_grid.py` -> `f_nl_grid_3x5`
* **Input:** the eigenbasis `eigs_basis` (sha256 2bb42868...). Rows are l=0,1,2 and columns n=0..4. r is
  in code units.
* **Plotting:** the toolkit function `fnl_plotting.plot_eigenfunction_panel` with its defaults. That
  function is the packaged form of `f_nl_python.ipynb` cell 9.
* **Differences:** none visible. The eigenbasis of the binary run (`eigs_basis_binary256`, sha256
  55f3b36a...) has identical eigenfunctions (to 5e-10). Its eigenvalues are shifted by a constant 826.53
  code units, which is a change of potential zero point.

### Fig. 4 (label `fig:3`): `fig04_equal_mass_binary.py` -> `0.2_Binary_pc_v2`
* **Run:** `binary256`.
* **Observable:** |x_BH1| from the origin, in pc. One curve is drawn, labelled "BH1 & BH2 (from Origin)",
  because the binary is symmetric: |r1 - r2| < 5e-8 pc.
* **Time axis:** legacy, as published (`--legacy-time-axis true`).
* **Style:** original notebook `Test_PyUL_alan_Binary.ipynb`, cell 27 with `PLOT_SINGLE_BH=True`, plus the
  kernel state from cell 5 (`axes.linewidth=2`).
* **Differences:** the saved cell 27 adds white/black path-effect outlines that the published PDF does not
  have. These are off by default (`--path-effects`). The published PDF is a Ghostscript conversion of the
  EPS with Helvetica fonts; where Helvetica is missing, Arial is substituted.

### Fig. 5 (label `fig:4`): `fig05_cnlm_3x3.py` -> `c_nlm_3x3_n012_v7`
* **Rows:** `cnlm_nobh128` (**N=128**), `cnlm_binary256` and `cnlm_single256_3000`.
* **Panels:** columns n=0,1,2, all l<=2 and m. |c|^2 / M_code on a log scale, y from 1e-13 to 50, window
  0-1500 Myr.
* **Time sampling:** as published, each 501-sample file is resampled to 60001 points with
  `c_nlm_integrator.resample_cnlm_time(..., 'nearest-savgol')`. This holds the nearest complex sample and
  Savitzky-Golay-smooths |c|, and is display-only. `--time-sampling original` plots the saved samples.
* **Style:** original notebook `compute_c_nlm_updated.ipynb`, cell 15 (kept with the `nobh128` run).
* **Log axis ticks:** the y major ticks are pinned to the published positions (every second decade from
  1e-13), because Matplotlib 3.11 (unlike 3.9.3, which made the published figure) picks a different
  stride. `--y-major-ticks auto` restores Matplotlib's own choice.
* **Differences:** fonts only. Validation for this figure was visual only.

### Figs. 6 and 7 (labels `fig:5`, `fig:6`): `fig06_fig07_cnlm_single_modes.py` -> `cnlm_n1_l1_mneg1_original_v3` (Fig. 6), `cnlm_n1_l0_m0_original_v3` (Fig. 7)
* **Input:** `cnlm_single256_3000`, using the saved samples (0-3000 Myr) and |c|^2 / M_code.
* **Plotting:** the toolkit function `cnlm_spectral_analysis.plot_cnlm_prd_single`, with (1,1,-1) in green
  (Fig. 6) and (1,0,0) in red (Fig. 7) as published.
* **Differences:** none. The saved notebook cell hard-codes green; the red l=0 version was exported after
  editing the colour.

### Fig. 8 (panels a-d; labels `fig:8a`, `fig:8b`, `fig:8c`, `fig:8f`): `fig08_reconstructed_background.py` -> `stone_skipping_{compare_n0,compare_n1,compare_n2,collection2}_pc_v2`
* **Runs:** `recon_nr3_nd*` and `recon_nr0_nd*`, the N=128 reduced model with DF A=0.75. Panels (a), (b),
  (c) are `compare_n0`, `compare_n1`, `compare_n2`; panel (d) is `collection2` (label `fig:8f`).
* **Observable:** |x_BH - x_ULDCOM| in pc, from the original notebook `plot.ipynb`, cell 5.
* **Time axis:** legacy, as published.
* **Legend font:** 14 pt for the compare panels and 10 pt for collection2, as published. The saved
  notebook holds only 10.
* **Differences:** fonts only. `--panels collection1 compare_n3` also draws the two unpublished panels
  (they need the supplementary runs `recon_nr3_ndnone` and `recon_nr3_nd3`).

### Fig. 9 (label `fig:dipole_removed`): `fig09_higher_mode_test.py` -> `distance_3d_1x2_pc_v3`
* **Runs:** `byhand_all` and `byhand_no_l1`.
* **Observable:** 3-D |x_BH - x_ULDCOM| in pc. `--quantity absz|rxy` gives the notebook's other exports.
* **Time axis:** legacy, and the x-limits are [t_0, t_last] = [0, 2994.01], as published.
* **Style:** original notebook `Test_PyUL_alan_2.ipynb`, cell 31.

### Figs. 10 and 11: `fig10_reduced_seeded_binary.py`, `fig11_fullsp_seeded_binary.py`
* **Published:** `equal_mass_reduced_seeded_vs_unseeded` (Fig. 10, label `fig:equal_mass_reduced_seeded`) and
  `equal_mass_seeded_vs_unseeded_fullSP` (Fig. 11, label `fig:equal_mass_seeded_fullSP`).
* **How the published curves were made:** they were digitised with PyMuPDF from older figure PDFs. The
  digitiser scripts survive only in an archived working-session log of 2026-06-24 (not deposited). These
  scripts read the raw trajectories instead, using the observable of the source PDFs: |x_BH| from the
  origin, in pc (original notebook `Test_PyUL_alan_Binary.ipynb`, cells 25 and 27).
* **Unseeded curve:** BH1 of `binary256`, which is **N=256**.
* **Seeded runs:** `binary2_reduced_seeded` (Fig. 10) and `binary2_fullsp_seeded` (Fig. 11), both N=128.
* **Seeded A/B mapping:** follows the drawing order of the source PDFs. In Fig. 10, A = BH1 and B = BH2.
  In Fig. 11, A = BH2 and B = BH1. Both can be changed with `--seeded-a-body`/`--seeded-b-body`.
* **Time axis:** legacy, as in the source PDFs.
* **Style:** the digitiser's `plot()` function.
* **Differences:** the published curves are digitised and keep only 320-416 of the 501 samples per curve.
  They agree with the raw data to within 0.1 pc (see Validation).

### Fig. 12 (label `fig:14`): `fig12_cnlm_time_spectrum.py` -> `cnlm_time_spectrum_n1_ell1_m-1_v3`
* **Input:** `cnlm_single256_30000` (N=256, 30000 Myr, 500 saves). The left panel shows 0-10000 Myr; the
  FFT uses the whole series. This is the same simulation (run ID `20260119_062714@256`) as the N=256 curve of
  Fig. 2 (`res256`); the deposit holds two copies of it under the two run keys.
* **Analysis and plotting:** the toolkit functions `analyze_cnlm_spectrum` (complex mode, Hann window,
  mean removed, `peak_rel_threshold=0.01`) and `export_prd_figure(time_xlim=(0, 10000))`.
* **Time sampling:** as published, the 60001-point nearest-savgol resampling. Its FFT bins give exactly
  the quoted f1 = 9.99983e-4 and f2 = -2.33329e-4 Myr^-1. `--time-sampling original` uses the 501
  samples, which give 9.98004e-4 and -2.32868e-4 (both within the frequency resolution 3.3e-5 Myr^-1 of the
  30000 Myr series; errata E6).
* **Known issue:** this run is PyUL 30.38 (the same simulation as `res256`), so its a_lm radii are 0.39 % too
  large (errata E4). The frequencies are unaffected; the amplitudes change by about 1 % when corrected.

### Fig. 13 (label `fig:16`): `fig13_resonance_roots.py` -> `f_x`
* **Content:** analytic. mu_pm / kappa^2 from Eq. (B77) of the printed paper (label `eq:C85`) is drawn where
  it is real. The vertical lines are
  at x = 1/sqrt(8 +- 4 sqrt3) = 0.258819 and 0.965926.
* **Source:** the original script was not found (the PDF was made with Matplotlib 3.7.5). The style was
  rebuilt from the published PDF's vector content: size, colours, line widths, limits and label positions.
* **Differences:** label positions are approximate. The axis label "f(x)" and the legend f_pm are kept as
  published.

## Validation (2026-10-18)

* **Fig. 1:** radii equal `derived/fig1_256_series_20261008.json` exactly (max |diff| = 0.0 pc; times
  identical). They match the published vector paths to 1.4e-4 pc on the PyUL time axis; with the legacy
  axis the mismatch is up to 5.3 pc.
* **Fig. 2:** N=192 and N=256 r_sys and P_dip recomputed from NBody and the `c_nlm` files equal the
  published CSV (0.0 pc; 1.7e-18). The published N=128 column equals the `res128` rerun (c_nlm sha256
  9a2f7851...): differences are 0.0 pc and 2.6e-18.
* **Figs. 4, 8 and 9:** the legacy axis matches the published vector paths to 0.0015-0.024 pc. The PyUL
  axis misses by 1.4-21 pc. All published curves end at t = 2994.01 Myr.
* **Figs. 10 and 11:** the raw data on the legacy axis matches the digitised published curves to <= 0.10
  pc. Swapping the BH assignment misses by about 150 pc.
* **Figs. 6 and 7:** match the published paths to 9e-9 (|c|^2 / M).
* **Fig. 12:** the left-panel |c| matches the dense series to 1.3e-7. The saved samples miss by 1e-3.
  The peaks equal the manuscript values to within 4.4e-10 Myr^-1.
* **Fig. 13:** matches the published curves to 7e-4.
* **Layouts:** every figure made from the deposit (bundles) is identical to the same figure made from the
  raw run directories: same plotted arrays, pixel-identical PNG, PDF/EPS identical apart from the creation
  date (Python 3.9.25/Matplotlib 3.9.3 and Python 3.13/Matplotlib 3.11).
* **Visual checks:** every figure was compared side by side with the published PDF. Remaining differences
  are fonts, where Helvetica/STIX/DejaVu are substituted.

## Tests

`pytest tests/test_trajectories.py tests/test_paper_figures.py` runs the synthetic-run unit tests and the
CLI checks. The data checks (Figs. 1, 2, 8 and 12, and `make_all_figures.py` on the deposit) also run when
`ULDM_PAPER_DATA_ROOT` points to the data root.
