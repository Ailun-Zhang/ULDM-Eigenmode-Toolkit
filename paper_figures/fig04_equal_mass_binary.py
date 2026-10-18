#!/usr/bin/env python3
"""Fig. 4 (tex label fig:3): orbital radius of the unseeded equal-mass binary (two 2 % BHs), fully coupled, N=256.

Source run: binary256 (run ID 20251022_031402@256).
Observable: |x_BH| from the simulation origin, pc (Test_PyUL_alan_Binary.ipynb cell 25
``distance_from_origin``).  Because the binary is symmetric the published figure draws one curve,
BH1, labelled "BH1 & BH2 (from Origin)" (cell 27 with PLOT_SINGLE_BH=True, USE_BW_STYLE=False).
Time axis: the published figure used the notebooks' compressed axis k*3000/(500+1) Myr (last point at
2994.01 Myr, verified from the PDF vector paths), so ``--legacy-time-axis`` defaults to true here.
Style: Test_PyUL_alan_Binary.ipynb cell 27 (figsize 7.2 x 4.6, black 2.4 pt line), plus the kernel state
left by cell 5 of that notebook (axes.linewidth = 2), which the published frame shows.  The saved cell 27
adds white/black path-effect outlines; the published PDF has a single 2.4 pt stroke (no outlines), so
``--path-effects`` defaults to false.
"""

from __future__ import annotations

import numpy as np

import _common


def compute(ctx, legacy_time_axis=True):
    run = ctx.run("binary256", legacy_time_axis=legacy_time_axis)
    return {"t": run.times_myr, "r1_pc": run.distance_from_origin_pc(0), "r2_pc": run.distance_from_origin_pc(1)}


def plot(d, args):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    plt.rcParams.update({"axes.linewidth": args.axes_linewidth})  # notebook cell 5 kernel state
    plt.rcParams.update({
        "font.size": 16, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "mathtext.fontset": "stix", "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans L", "Liberation Sans", "Arial", "DejaVu Sans"],
        "ps.useafm": True, "ps.fonttype": 3, "axes.unicode_minus": False,
    })
    t, r1_pc, r2_pc = np.asarray(d["t"]), d["r1_pc"], d["r2_pc"]
    r_single = r2_pc if args.single_series == "bh2" else r1_pc
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if args.style == "bw":
        if args.plot_single_bh:
            ax.plot(t, r_single, color="#333333", linestyle="-", linewidth=2.4, label=args.single_label)
        else:
            ax.plot(t, r1_pc, color="#333333", linestyle="-", linewidth=2.4, label="BH1 (from Origin)")
            ax.plot(t, r2_pc, color="#777777", linestyle="--", linewidth=2.8, label="BH2 (from Origin)")
    else:
        marker_step = max(1, len(t) // 16)
        mark_bh1 = list(range(0, len(t), marker_step))
        mark_bh2 = list(range(marker_step // 2, len(t), marker_step))
        if args.plot_single_bh:
            l1, = ax.plot(t, r_single, color="black", linewidth=2.4, label=args.single_label, zorder=3,
                          solid_capstyle="round", markevery=mark_bh1, markersize=4.0, markerfacecolor="white",
                          markeredgewidth=1.0, markeredgecolor="#111111")
            if args.path_effects:
                l1.set_path_effects([pe.Stroke(linewidth=l1.get_linewidth() + 1.6, foreground="white"),
                                     pe.Stroke(linewidth=l1.get_linewidth() + 0.6, foreground="#111111"),
                                     pe.Normal()])
        else:
            # cell 27 as saved: BH1 drawn with alpha=0 (invisible) but kept in the legend
            l2, = ax.plot(t, r2_pc, color="#d62728", alpha=1, linewidth=3.0, label="BH2 (from Origin)", zorder=2,
                          solid_capstyle="round", markevery=mark_bh2, markersize=8.0, markerfacecolor="white",
                          markeredgewidth=1.0, markeredgecolor="#d62728")
            l1, = ax.plot(t, r1_pc, color="black", alpha=args.bh1_alpha, linewidth=2.4, label="BH1 (from Origin)",
                          zorder=3, solid_capstyle="round", markevery=mark_bh1, markersize=4.0,
                          markerfacecolor="white", markeredgewidth=1.0, markeredgecolor="#111111")
            for line in ((l1, l2) if args.path_effects else ()):
                line.set_path_effects([pe.Stroke(linewidth=line.get_linewidth() + 1.6, foreground="white"),
                                       pe.Stroke(linewidth=line.get_linewidth() + 0.6, foreground="#111111"),
                                       pe.Normal()])
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Distance from Origin (pc)")
    ax.set_title("")
    ax.grid(False, which="both")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = _common.make_parser("fig04", __doc__.splitlines()[1])
    parser.add_argument("--legacy-time-axis", type=_common.str2bool, default=True,
                        help="old-notebook axis k*T/(save_number+1) (published: true)")
    parser.add_argument("--plot-single-bh", type=_common.str2bool, default=True,
                        help="draw one curve for the symmetric binary (published: true)")
    parser.add_argument("--single-series", choices=("bh1", "bh2"), default="bh1")
    parser.add_argument("--single-label", default="BH1 & BH2 (from Origin)")
    parser.add_argument("--style", choices=("color", "bw"), default="color",
                        help="cell 27 USE_BW_STYLE=False ('color', published) or True ('bw')")
    parser.add_argument("--path-effects", type=_common.str2bool, default=False,
                        help="white/black outline strokes of the saved cell 27 (absent in the published PDF)")
    parser.add_argument("--axes-linewidth", type=float, default=2.0,
                        help="frame width (published: 2, from the notebook's cell 5 rcParams)")
    parser.add_argument("--bh1-alpha", type=float, default=0.0,
                        help="alpha of the BH1 line when both bodies are drawn in 'color' style (cell 27: 0)")
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args)
    d = compute(ctx, args.legacy_time_axis)
    fig = plot(d, args)
    paths = ctx.save(fig, {})  # cell 27: fig.savefig(path, format=...) with rcParams savefig.bbox='tight'
    ctx.write_provenance({"observable": "|x_BH| from simulation origin [pc]",
                          "max_abs_bh1_minus_bh2_pc": float(np.max(np.abs(d["r1_pc"] - d["r2_pc"])))})
    for p in paths:
        print(p)
    return d


if __name__ == "__main__":
    main()
