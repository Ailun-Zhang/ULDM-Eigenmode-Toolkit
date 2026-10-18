#!/usr/bin/env python3
"""Fig. 11 (fig:14): c_{1,1,-1}(t) in the time domain (|c|, Re, Im) and its FFT spectrum with the two peaks.

Source: cnlm_single256_30000 (c_nlm_mass170.h5 of the single-BH run 20260119_062714@256),
a 30000 Myr / 500-save N=256 single-BH run.  Normalisation c / sqrt(M_code), M_code = 12e6 Msun in code units.
Analysis and plotting: the toolkit functions cnlm_spectral_analysis.analyze_cnlm_spectrum (complex mode, Hann
window, mean removed, peak_rel_threshold 0.01) and export_prd_figure (left panel cropped to 0-10000 Myr), as in
the toolkit notebook compute_c_nlm.ipynb, cells 24/26, run on this file (original copy 'compute_c_nlm copy.ipynb').
Time sampling: the published spectrum was computed from the 60001-point "nearest-savgol" resampling of the
whole 0-30000 Myr series (use_dense=True): its peaks sit exactly on that grid's FFT bins
(f1 = 30/30000.5 Myr^-1 = 9.99983e-4, f2 = -7/30000.5 = -2.33329e-4, the values quoted in the text).
``--time-sampling original`` uses the 501 saved samples instead (bin width 1/30060 Myr^-1).
"""

from __future__ import annotations

import contextlib

import numpy as np

import _common


@contextlib.contextmanager
def capture_show(callback):
    import matplotlib.pyplot as plt

    original = plt.show

    def _show(*a, **k):
        callback(plt.gcf())

    plt.show = _show
    try:
        yield
    finally:
        plt.show = original


def compute(ctx, args):
    from Eig_ULDM_packages.c_nlm_integrator import resample_cnlm_time
    from Eig_ULDM_packages.cnlm_spectral_analysis import analyze_cnlm_spectrum

    key = tuple(args.mode)
    t_file, c_all, _ = _common.load_cnlm(ctx.file("cnlm_single256_30000"))
    if args.time_sampling == "dense":
        t = np.linspace(t_file[0], t_file[-1], int(args.n_dense), endpoint=True)
        c = resample_cnlm_time({key: c_all[key]}, t_file, t, method="nearest-savgol")[key]
    else:
        t, c = t_file, c_all[key]
    if args.analysis_t_max is not None:
        sel = t <= args.analysis_t_max
        t, c = t[sel], c[sel]
    m_code = _common.soliton_mass_code(ctx, "cnlm_single256_30000")
    result = analyze_cnlm_spectrum(t_data=t, c_data=c, key=key, analysis_mode=args.analysis_mode,
                                   normalization=m_code, apply_hann_window=True, detrend_mean=True,
                                   peak_rel_threshold=args.peak_rel_threshold, top_k_peaks=6)
    return result, m_code


def main(argv=None):
    parser = _common.make_parser("fig11", __doc__.splitlines()[1])
    parser.add_argument("--mode", type=int, nargs=3, default=[1, 1, -1], metavar=("N", "L", "M"))
    parser.add_argument("--time-sampling", choices=("dense", "original"), default="dense",
                        help="dense = 60001-point nearest-savgol resampling (published); original = 501 saved samples")
    parser.add_argument("--n-dense", type=int, default=60001)
    parser.add_argument("--analysis-t-max", type=float, default=None,
                        help="truncate the series before the FFT (published: no truncation, 0-30000 Myr)")
    parser.add_argument("--analysis-mode", choices=("complex", "real", "imag"), default="complex")
    parser.add_argument("--peak-rel-threshold", type=float, default=0.01)
    parser.add_argument("--time-xlim", type=float, nargs=2, default=(0.0, 10000.0))
    parser.add_argument("--figsize", type=float, nargs=2, default=(7.6, 3.1))
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args)

    from Eig_ULDM_packages.cnlm_spectral_analysis import export_prd_figure
    import matplotlib.pyplot as plt

    result, m_code = compute(ctx, args)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    paths = ctx.output_paths()
    eps_path = paths.get("eps", ctx.out_dir / f"{ctx.basename}.eps")

    def save_other(fig):
        for fmt, path in paths.items():
            if fmt in ("eps", "pdf"):
                continue  # the toolkit writes EPS and PDF itself (bbox_inches='tight')
            kw = {"bbox_inches": "tight"}
            if fmt == "png":
                kw["dpi"] = ctx.png_dpi_for()
            fig.savefig(path, format=fmt, **kw)

    with capture_show(save_other):
        export_prd_figure(result, save_path=eps_path, figsize=tuple(args.figsize),
                          time_xlim=tuple(args.time_xlim) if args.time_xlim else None,
                          save_pdf="pdf" in ctx.formats)
    plt.close("all")
    if "eps" not in ctx.formats and eps_path.exists():
        eps_path.unlink()
    ctx.written.extend(str(p) for p in paths.values())
    peaks = sorted(result.significant_peaks, key=lambda z: -z[1])
    info = {"M_code": m_code, "N_fft": result.N, "dt_myr": result.dt_med, "Tspan_myr": result.Tspan,
            "peak_pattern": result.peak_pattern, "significant_peaks": peaks,
            "f_env": result.f_env, "fft_note": result.fft_note}
    if len(result.strong_peaks_2) == 2:
        f_a, f_b = sorted(p[0] for p in result.strong_peaks_2)
        info["f1_f2_delta"] = [f_b, f_a, f_b - f_a]
    ctx.write_provenance(info)
    for p in paths.values():
        print(p)
    print(json_line(info))
    return result


def json_line(info):
    import json

    return json.dumps({k: info[k] for k in ("peak_pattern", "significant_peaks", "f_env") if k in info},
                      default=float)


if __name__ == "__main__":
    main()
