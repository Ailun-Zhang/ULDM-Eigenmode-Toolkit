# Conventions and errata of the published analysis

This page documents the numerical conventions of the analysis behind *Stone Skipping Black Holes in
Ultralight Dark Matter Solitons* (A. Zhang, Y. Wang, J. L. Zagorac and R. Easther,
[arXiv:2602.11512](https://arxiv.org/abs/2602.11512)) and the known issues of the code versions that
produced its figures. The paper itself does not discuss them; this page and the [CHANGELOG](../CHANGELOG.md)
do.

Figure numbers are those of the printed paper (13 figures; Figs. 6 and 7 are the two single-mode panels
made by `paper_figures/fig06_fig07_cnlm_single_modes.py`). Where a LaTeX label is quoted, it is marked as a
label.

Summary:

| ID | Item | Published figures concerned | Effect | Toolkit 0.3.0 |
|---|---|---|---|---|
| E1 | legacy angular grid of the a_lm kernel | every c_nlm figure: 2, 5, 6, 7, 12 | l >= 1 amplitudes biased by 0.1-0.4 %, leakage 0.3-4 % between (l, m); frequencies unaffected | default kept; `angular_grid='dh'` |
| E2 | eigen-solver wall at r_max + h | every c_nlm figure, Fig. 3 | eigenvalues depend on the grid at the 0.25 % level (5120 vs 10240 points) | default kept; `outer_boundary='rmax'` |
| E3 | rounded soliton `mass0` | every c_nlm figure, Fig. 3 | scaled soliton about 6e-5 too light | default kept; `mass0=` / `--soliton-mass0` |
| E4 | PyUL 30.38/30.40 resampled box (`Resample3Box`) | **Fig. 2** (all three resolutions) and **Fig. 12** | a_lm radii labelled 0.39-0.79 % too large; P_dip -0.6 % to -2.4 %, \|c_000\|^2/M -1.2 % to -2.4 % when corrected; frequencies unaffected | documented; PyUL_SK 30.50 fixed by default |
| E5 | legacy time axis k T / (save_number + 1) | **Figs. 4, 8, 9** and the digitised sources of **Figs. 10, 11** | time axis compressed by 500/501 (last point at 2994.01 Myr instead of 3000 Myr) | reproduced by default in the figure scripts; `legacy_time_axis=False` |
| E6 | dense "nearest-savgol" display series | **Figs. 5 and 12** | smoothed \|c\| on a 60001-point grid; FFT peaks differ from the 501-sample values by less than the frequency resolution | reproduced by default; `--time-sampling original` |

None of these changes a qualitative conclusion of the paper. The defaults of the toolkit reproduce the
published numbers (policy B of the changelog); the switches below give the corrected variants.

---

## E1. Angular grid of the a_lm kernel (legacy, not Driscoll-Healy)

**What.** `functions3.compute_sh_coeffs` / `process_file` (and the one-pass kernel `alm_kernel`, which is
bitwise equal to them) sample each shell on

```
theta = linspace(0, pi, N)        # both endpoints
phi   = linspace(0, 2 pi, 2N)     # both endpoints
```

and pass the samples to `pyshtools.expand.SHExpandDHC(norm=4, sampling=2, csphase=-1)`, which expects the
Driscoll-Healy grid `theta_i = pi i / N`, `phi_j = pi j / N` (no closing endpoints). The paper used N = 300
and l_max = 10.

**Size (N = 300, pure Y_lm in, legacy grid; the 'dh' grid is exact to 1e-15):**

| input Y_lm | \|a_lm\| - 1 | arg a_lm [rad] | largest leakage into another (l', m') |
|---|---|---|---|
| (0,0) | -1e-16 | 0 | 4e-16 |
| (1,0) | +2.2e-3 | 0 | 7.1e-3 |
| (1,1) | -1.1e-3 | +5.2e-3 | 3.5e-3 |
| (2,0) | +3.1e-3 | 0 | 1.2e-2 |
| (2,2) | -1.4e-3 | +1.05e-2 | 5.1e-3 |
| (5,3) | -9.3e-4 | +1.57e-2 | 1.4e-2 |
| (10,0) | +3.8e-3 | 0 | 4.0e-2 |
| (10,10) | -2.1e-3 | +5.2e-2 | 1.6e-2 |

The phase offset is m pi / (2N - 1). On a real 64^3 PyUL snapshot the radially integrated power per l changes by
2.5e-5 (l = 0) and by 0.03-1.3 % (l = 1 ... 10) between the two grids.

**Effect on the paper.** The distortion is linear and fixed in time, so it is the same for every snapshot
and every run: c_nlm frequencies (Fig. 12) are unaffected, and l >= 1 amplitudes (the dipole occupation of
Fig. 2, the l = 1 curves of Figs. 5 and 6) are biased at the 0.1-1 % level. Y_00 itself is sampled exactly,
so l = 0 coefficients (Fig. 7, the l = 0 curves of Fig. 5) change only through leakage from l >= 1 structure.

**Switch.** `angular_grid='dh'` in `compute_alm_cube`, `AlmKernel`, `functions3.process_file`,
`compute_sh_coeffs`, `uldm-cnlm --angular-grid dh`, `compute_coeff.py --angular-grid dh` and PyUL_SK's
`CnlmAngularGrid='dh'`. Default: `'legacy'`.

## E2. Eigen-solver outer boundary at r_max + h

**What.** `uldm_eig1.solve_h` builds the finite-difference Hamiltonian on `r = linspace(0, r_max, n_grid)`,
drops r = 0 and keeps the last grid point r_max as an unknown. The Dirichlet wall is therefore effectively at
r_max + h (h = r_max / (n_grid - 1)), not at r_max.

**Size.** On the analytic free sphere the legacy boundary gives a first-order bias of about 2h/R in the
eigenvalues. For the paper basis (n_grid = 5120 including r = 0, r_max = 2.5 kpc, lmax 10, 25 states per l)
an audit on the production cluster measured an energy change of about 0.25 % between 5120 and 10240 points,
against about 1e-5 for the fixed-domain scheme. The basis is the same for every run and snapshot, so the
effect is common to all published c_nlm and does not change their frequencies.

**Switch.** `outer_boundary='rmax'` (`u(r_max) = 0` exactly; converges at second order) in `solve_h` and
`make_all_eigs`, `uldm-cnlm --eigen-outer-boundary rmax`, PyUL_SK `CnlmEigenParams`. Default: `'legacy'`.

## E3. Rounded soliton mass in `soliton.h5`

**What.** The bundled `Eig_ULDM_packages/soliton.h5` (its `/profile` dataset is PyUltraLight's standard
profile `initial_f.npy`, bit for bit) stores `mass0 = 3.883`. The integral of the profile is
3.8827652755822006, which PyUL uses for its solitons.

**Size.** The soliton scaled to a requested mass with the rounded value is about 6e-5 (relative) too light,
which changes the soliton potential and its eigenbasis at the 1e-4 level, far below every effect discussed in
the paper.

**Switch.** `load_soliton_file(path, mass0=3.8827652755822006)`, `uldm-cnlm --soliton-mass0`. Default: the
file value (all published eigenfunctions, Fig. 3). `soliton.h5` itself is unchanged.

## E4. PyUL 30.38 / 30.40 resampled box: a_lm radii of Fig. 2 and Fig. 12

**What.** The c_nlm files are computed from `Outputs/3WfnRS/P3R_#NNN.npy`, the wavefunction resampled by
PyUL's `Resample3Box` onto a `resolR`^3 cube centred on the centre of mass (paper: resolR = 256,
Length_Ratio = 1, 5 kpc box). On cell-centred grids (`CellCenteredGrid=True`, used by the paper runs) PyUL
30.38 and 30.40 resampled onto a side of `(L - L/N) * Length_Ratio` instead of `L * Length_Ratio`, where N is
the simulation resolution. The a_lm scripts (`compute_coeff.py`, the notebooks) assumed the spacing
`L / resolR`. The true spacing is smaller by 1/N, so each shell labelled r in the a_lm and c_nlm files lies
at r (1 - 1/N): the labels are too large by 1/(N - 1) = 0.79 %, 0.52 % and 0.39 % at N = 128, 192 and 256.
The 26.32 cores used the correct side.

**Which runs and figures.** The value of |c_000|^2 / M at t = 0, which is 1 for an exactly labelled soliton,
identifies the affected files:

| run key | PyUL | N | \|c_000\|^2/M at t = 0, as published | recomputed with the true spacing |
|---|---|---|---|---|
| all 26.32 runs (`single256_3000`, `binary256`, `nobh128`, `res128_jan2026`) | 26.32 | 128, 256 | 0.994 - 1.000 | (not affected) |
| `res256` = `single256_30000` | 30.38 | 256 | 1.0096 | 0.998 |
| `res192` | 30.40 | 192 | 1.0132 | 0.997 |
| `res128` | 30.40 | 128 | 1.0191 | 0.995 |

* **Affected:** printed **Fig. 2** (all three resolution runs: P_dip in panel b) and printed **Fig. 12** (the
  spectrum of c_{1,1,-1}; its 30000 Myr, 256^3 run is the same simulation as `res256`, PyUL 30.38).
* **Not affected:** printed Figs. 5, 6 and 7 (PyUL 26.32 runs), Fig. 3 (eigenbasis only) and every
  trajectory figure (the dynamics never use the resampled cube).

**Size.** The three affected runs were recomputed with the true spacing (one-pass kernel, same eigenbasis,
legacy angular grid, snapshots at 0, 1500 and 3000 Myr; cluster job 21987):

| run | P_dip (sum over n <= 2, m of \|c_n1m\|^2 / M) | \|c_000\|^2 / M | \|c_{1,1,-1}\|^2 / M at 1500 / 3000 Myr |
|---|---|---|---|
| 128^3 (`res128`, 30.40) | -1.6 % to -2.4 % | -2.3 % to -2.4 % | +0.8 % / -2.6 % |
| 192^3 (`res192`, 30.40) | -0.9 % to -1.6 % | -1.6 % | +0.4 % / -1.5 % |
| 256^3 (`res256`, 30.38) | -0.6 % to -1.2 % | -1.2 % | -0.1 % / -1.0 % |

So, when corrected, P_dip changes by -0.6 % to -2.4 % and |c_000|^2 / M by -1.2 % to -2.4 %, the largest
changes at 128^3; the corrected |c_000|^2/M agree with the 26.32 runs. The radius error is a time-independent
linear distortion, so the frequencies of Fig. 12 are unaffected and its amplitudes change by about 1 %. In
Fig. 2 the spread between resolutions shrinks only slightly (the 128^3 - 256^3 difference in P_dip at
1500 Myr goes from about 13.8 % to 12.7 %). The qualitative conclusions are unchanged. The difference of
about 3.5 % in P_dip between the two 128^3 runs (`res128`, 30.40, and the supplementary `res128_jan2026`,
26.32) is partly this effect.

Radial (l = 0, n >= 1) coefficients of these four files are much more sensitive than the figures: a
soliton whose radii are mislabelled projects onto the radial breathing mode, so |c_100|^2 / M at t = 0 drops
from 7.5e-5 to 2.5e-6 (128^3) when corrected, and changes by +10 % to +32 % at 1500 and 3000 Myr. No printed
figure shows l = 0, n >= 1 modes of these runs; use the 26.32 runs, or recompute, for such work.

**How to avoid it.**
* New runs: PyUL_SK 30.50 resamples onto `L * Length_Ratio` by default and records the geometry in
  `reproducibility.uldm` (`"SK Grid and Output Settings"` -> `"Resampled Box"`), which `uldm-cnlm` and the
  inline recorder read. `evolve(..., LegacyResampleBox=True)` reproduces the 30.38/30.40 cube bit for bit.
* Existing 30.38/30.40 runs: pass the true geometry to the pipeline, e.g. for N = 128
  `uldm-cnlm RUN --rs-side 4.9609375 --rs-resol 256 --build-eigen` (side `(L - L/N) * Length_Ratio` in kpc),
  or `compute_coeff.py --rs-length-kpc 4.9609375 --rs-resol 256`.
* The deposited c_nlm files are the published ones and are not corrected.

## E5. Legacy time axis of Figs. 4, 8, 9 (and the sources of Figs. 10, 11)

PyUL saves frame k at t_k = start + k * duration / save_number. The analysis notebooks that drew printed
**Figs. 4, 8 and 9** used `t_k = k * duration / (save_number + 1)`, and so did the older figure PDFs from which
the curves of printed **Figs. 10 and 11** were digitised. For 500 saves over 3000 Myr this compresses the
axis by 500/501: the last point is drawn at 2994.01 Myr instead of 3000 Myr (a shift of up to 6 Myr, about one
save interval). This was verified from the vector paths of the published PDFs (the legacy axis matches them to
0.0015 - 0.024 pc, the PyUL axis misses by 1.4 - 21 pc). Figs. 1, 2, 5, 6, 7 and 12 use the correct axis.

The data themselves (bundles, c_nlm files) carry the correct PyUL times. The figure scripts reproduce the
published axis by default (`--legacy-time-axis true` for Figs. 4, 8, 9, 10 and 11);
`--legacy-time-axis false` and `trajectories.load_run(path)` give the PyUL axis, and
`trajectories.load_run(path, legacy_time_axis=True)` the legacy one.

## E6. Dense "nearest-savgol" series of Figs. 5 and 12

Printed **Figs. 5 and 12** were drawn from a 60001-point resampling of each 501-sample c_nlm file, made with
`c_nlm_integrator.resample_cnlm_time(c, t, t_dense, method='nearest-savgol')`:

1. nearest-neighbour hold of the complex sample (the phase is piecewise constant);
2. Savitzky-Golay smoothing of |c| (window 2 * 120 + 1 = 241 points, polynomial order 2 for 501 -> 60001);
3. the smoothed modulus is put back with the held phase.

This is display smoothing, not new information. The FFT of Fig. 12 was taken on the dense series of the
30000 Myr run; its peaks are f1 = 9.99983e-4 and f2 = -2.33329e-4 Myr^-1, while the 501 saved samples give
9.98004e-4 and -2.32868e-4 Myr^-1. Both agree within the frequency resolution 1/(30000 Myr) = 3.3e-5 Myr^-1, so
f1 = 1.00e-3 and f2 = -2.33e-4 Myr^-1 are significant to about that resolution.

The dense files (about 2.9 GB per run) are not deposited; the figure scripts recompute them
(`--time-sampling dense`, default) or plot the saved samples (`--time-sampling original`). Since 0.3.0,
`resample_cnlm_time(method='nearest')` is a pure nearest-neighbour hold; it applied the Savitzky-Golay step
silently before. The paper notebooks used `'nearest-savgol'` explicitly and are unaffected.

---

## Other conventions

* **c_nlm normalisation.** Raw c_nlm are in sqrt(code mass): `sum |c|^2` = mass in code units, and a pure
  soliton has c_000 = sqrt(M). The figures plot |c|^2 / M_code with M_code = 1.2e7 Msun in code units
  = 170.3149822613627 at m_a = 1e-21 eV.
* **Axion mass.** The paper used m_a = 1e-21 eV (m22 = 10). `units` defaults to 1e-22 eV unless
  `PYUL_AXION_MASS_EV` is set or `units.set_axion_mass` is called; the run readers take the mass recorded in
  `reproducibility.uldm`.
* **Eigenbasis zero point.** The two eigenbasis files of the data deposit have the same eigenfunctions (to
  5e-10) and eigenvalues offset by a constant 826.53 code units (a different potential zero point); c_nlm are
  unaffected.
* **Observables.** Figs. 1, 2, 4, 10, 11: |x_BH| from the simulation origin (the initial total centre of mass).
  Figs. 8, 9: |x_BH - x_ULDCOM|. Fig. 2b: P_dip = sum over n <= 2 and m of |c_n1m|^2 / M_code.

See also the "Known issues and conventions" section of the
[PyUL_SK README](https://github.com/Ailun-Zhang/PyUL_SK#known-issues-and-conventions) for the simulation side.
