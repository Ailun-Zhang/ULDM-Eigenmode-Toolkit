#!/usr/bin/env python3
"""Fig. 5 (fig:4): 3 x 3 panel of |c_{n l m}(t)|^2 / M_code, columns n = 0, 1, 2, all l <= 2 and m.

Rows (original notebook compute_c_nlm_updated.ipynb, cell 15, kept with the nobh128 run 20251016_071245@128):
  1. "No Black Hole (BH)"            cnlm_nobh128        (N=128!)
  2. "Binary no Stone Skipping (SK)" cnlm_binary256      (N=256)
  3. "Single BH with SK"             cnlm_single256_3000 (N=256)
Normalisation: |c|^2 / M_code with M_code = 12e6 Msun in code units (170.31 at 1e-21 eV).
Window: 0-1500 Myr (t_plot_max_myr = 1500), x padded by 4 %; y log-scale, shared range (1e-13, 50).
Time sampling: the published figure plotted the 60001-point "nearest-savgol" resampling of each
501-sample c_nlm file (c_nlm_mass170_dense_from_file.h5, made with
c_nlm_integrator.resample_cnlm_time(..., method='nearest-savgol')).  That resampling holds the complex
value of the nearest saved sample and Savitzky-Golay-smooths only |c|; it is display-only and is
reproduced here by default (``--time-sampling dense``).  ``--time-sampling original`` plots the saved
samples instead.
"""

from __future__ import annotations

import numpy as np

import _common

ROWS = [("cnlm_nobh128", "No Black Hole (BH)"),
        ("cnlm_binary256", "Binary no Stone Skipping (SK)"),
        ("cnlm_single256_3000", "Single BH with SK")]


def resample_dense(t_file, c_sub, n_dense):
    from Eig_ULDM_packages.c_nlm_integrator import resample_cnlm_time

    t_dense = np.linspace(t_file[0], t_file[-1], int(n_dense), endpoint=True)
    # Each series is resampled independently, so resampling the plotted subset is identical to
    # resampling the whole file (as the notebook did) and needs ~1 % of the memory.
    return t_dense, resample_cnlm_time(c_sub, t_file, t_dense, method="nearest-savgol")


def compute(ctx, args):
    keys = [(n, l, m) for n in args.n_cols for l in range(args.l_min, args.l_max + 1) for m in range(-l, l + 1)]
    datasets = []
    for file_key, label in ROWS:
        t_file, c_all, _ = _common.load_cnlm(ctx.file(file_key))
        c_sub = {k: c_all[k] for k in keys if k in c_all}
        if args.time_sampling == "dense":
            t, c = resample_dense(t_file, c_sub, args.n_dense)
        else:
            t, c = t_file, c_sub
        if abs(t_file[-1] - args.expected_t_end) > 1e-6 * args.expected_t_end:
            raise ValueError(f"{file_key}: c_nlm time axis ends at {t_file[-1]}, expected {args.expected_t_end} Myr")
        sel = np.ones_like(t, dtype=bool)
        if args.t_min is not None:
            sel &= t >= args.t_min
        if args.t_max is not None:
            sel &= t <= args.t_max
        m_code = _common.soliton_mass_code(ctx, file_key)
        datasets.append({"key": file_key, "label": label, "t": t[sel],
                         "y": {k: (np.abs(v[sel]) ** 2) / m_code for k, v in c.items()},
                         "M_code": m_code, "missing": [k for k in keys if k not in c_all]})
    return datasets


_LW_BY_ABS_M = {0: 2.2, 1: 1.5, 2: 1.0}


def plot(datasets, args):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.ticker import FixedLocator, LogFormatterSciNotation, MaxNLocator, ScalarFormatter

    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 12, "axes.labelsize": 15, "axes.titlesize": 15,
        "legend.fontsize": 12, "axes.linewidth": 1.2,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5, "xtick.minor.size": 3.0, "ytick.minor.size": 3.0,
        "mathtext.fontset": "stix", "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans L", "Liberation Sans", "Arial", "DejaVu Sans"],
        "ps.useafm": True, "ps.fonttype": 3, "axes.unicode_minus": False,
    })
    red, green_dark, green_light = to_rgba("red"), to_rgba("green"), to_rgba("#00fa08")
    purple = {0: to_rgba("#5b0f9e"), 1: to_rgba("#8b3fcf"), 2: to_rgba("#c89ef0")}

    def color_for(ell, m):
        if ell == 0:
            return red
        if ell == 1:
            return green_dark if m == 0 else green_light
        if ell == 2:
            return purple[min(abs(m), 2)]
        return to_rgba("gray")

    def lw(m):
        return _LW_BY_ABS_M.get(abs(m), 0.8)

    def ls(m):
        return "-" if m == 0 else ("--" if m > 0 else "-.")

    x_min = min(float(np.min(d["t"])) for d in datasets)
    x_max = max(float(np.max(d["t"])) for d in datasets)
    x_pad = 0.04 * max(1.0, x_max - x_min)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7.4, 10.0), sharex=False, sharey=False,
                             constrained_layout=False, gridspec_kw={"wspace": 0.00, "hspace": 0.00})
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.92)
    for j, n_val in enumerate(args.n_cols):
        axes[0, j].set_title(rf"$n={n_val}$", pad=6)
    xfmt = ScalarFormatter(useOffset=False)
    xfmt.set_scientific(False)
    for i, d in enumerate(datasets):
        for j, n_val in enumerate(args.n_cols):
            ax = axes[i, j]
            ax.grid(False)
            for ell in range(args.l_min, args.l_max + 1):
                for m in range(-ell, ell + 1):
                    key = (n_val, ell, m)
                    if key not in d["y"]:
                        continue
                    ax.plot(d["t"], np.clip(d["y"][key], 1e-40, None), color=color_for(ell, m),
                            linestyle=ls(m), linewidth=lw(m), solid_capstyle="round")
            ax.set_yscale("log")
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.xaxis.set_major_formatter(xfmt)
            ax.xaxis.get_offset_text().set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
            if j != 0:
                ax.tick_params(labelleft=False)
            if i != 2:
                ax.tick_params(labelbottom=False)
            ax.tick_params(which="both", direction="in", top=True, right=True)
            if j > 0:
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False)
            if i > 0:
                ax.spines["top"].set_visible(False)
                ax.tick_params(top=False)
        for j in range(3):
            axes[i, j].set_ylim(*args.ylim)
            if args.y_major_ticks == "published":
                # The published figure (Matplotlib 3.9.3 auto LogLocator) has major ticks at 10^-13, 10^-11,
                # ..., 10^1; newer Matplotlib picks a different stride, so pin the published positions.
                exps = np.arange(np.ceil(np.log10(args.ylim[0])), np.floor(np.log10(args.ylim[1])) + 1, 2)
                axes[i, j].yaxis.set_major_locator(FixedLocator(10.0 ** exps))
                axes[i, j].yaxis.set_major_formatter(LogFormatterSciNotation())
    fig.canvas.draw()
    for i in (0, 1):  # hide the 1e-13 label of the upper rows (cell 15 "FIX 3a")
        for label in axes[i, 0].yaxis.get_ticklabels():
            txt = label.get_text().strip()
            if "-13" in txt or "−13" in txt:
                label.set_visible(False)
    for j in (1, 2):  # hide the "0" label of columns 1-2 (cell 15 "FIX 3b")
        for label in axes[2, j].xaxis.get_ticklabels():
            if label.get_text().strip() == "0":
                label.set_visible(False)
    fig.supxlabel(r"Time $t$ (Myr)", y=0.06)
    fig.supylabel(r"$\left|c_{n\ell m}(t)\right|^{2}$", x=0.01)
    for i, d in enumerate(datasets):
        bb = axes[i, 2].get_position()
        fig.text(0.998, 0.5 * (bb.y0 + bb.y1), d["label"], rotation=-90, va="center", ha="center", fontsize=12)
    legend_colors = {0: red, 1: green_dark, 2: purple[0]}
    ell_handles = [mpl.lines.Line2D([0], [0], color=legend_colors[ell], lw=1.8, linestyle="-")
                   for ell in range(args.l_min, args.l_max + 1)]
    m_handles = [mpl.lines.Line2D([0], [0], color="black", linestyle="-", lw=_LW_BY_ABS_M[0]),
                 mpl.lines.Line2D([0], [0], color="black", linestyle="--", lw=_LW_BY_ABS_M[1]),
                 mpl.lines.Line2D([0], [0], color="black", linestyle="-.", lw=_LW_BY_ABS_M[1]),
                 mpl.lines.Line2D([0], [0], color="black", linestyle="--", lw=_LW_BY_ABS_M[2]),
                 mpl.lines.Line2D([0], [0], color="black", linestyle="-.", lw=_LW_BY_ABS_M[2])]
    labels = [rf"$\ell={ell}$" for ell in range(args.l_min, args.l_max + 1)] + \
        ["m=0", "m=+1", r"m=$-$1", "m=+2", r"m=$-$2"]
    fig.legend(ell_handles + m_handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.085), frameon=False,
               ncol=8, handlelength=1.7, handletextpad=0.55, columnspacing=0.9, labelspacing=0.30)
    return fig


def main(argv=None):
    parser = _common.make_parser("fig05", __doc__.splitlines()[1])
    parser.add_argument("--time-sampling", choices=("dense", "original"), default="dense",
                        help="dense = 60001-point nearest-savgol resampling (published); original = saved samples")
    parser.add_argument("--n-dense", type=int, default=60001)
    parser.add_argument("--n-cols", type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument("--l-min", type=int, default=0)
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=1500.0)
    parser.add_argument("--expected-t-end", type=float, default=3000.0)
    parser.add_argument("--ylim", type=float, nargs=2, default=(1e-13, 50))
    parser.add_argument("--y-major-ticks", choices=("published", "auto"), default="published",
                        help="published = every second decade from 1e-13 (as printed); auto = Matplotlib's LogLocator")
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args)
    datasets = compute(ctx, args)
    fig = plot(datasets, args)
    paths = ctx.save(fig, {"bbox_inches": "tight", "pad_inches": 0.02})
    ctx.write_provenance({"normalisation": {d["key"]: d["M_code"] for d in datasets},
                          "missing_keys": {d["key"]: d["missing"] for d in datasets}})
    for p in paths:
        print(p)
    return datasets


if __name__ == "__main__":
    main()
