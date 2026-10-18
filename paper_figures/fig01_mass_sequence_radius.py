#!/usr/bin/env python3
"""Fig. 1 (tex label fig:1): orbital radius |x_BH| vs time for single BHs of 1, 1.5, 2, 3, 4, 5 % of the soliton mass.

Source runs: fig1_m1, fig1_m1p5, single256_3000 (2 %), fig1_m3, fig1_m4, fig1_m5 (N=256, 5 kpc box).
Observable: distance of the BH from the simulation origin (= initial total centre of mass), in pc;
no smoothing, alignment or offsets.  Time axis t_k = k * 3000 Myr / 500 (correct PyUL convention,
as in the published figure).  Style: the original figure script refine_figure1.py of the 2026-10-08 figure
update (overlay variant, the file published as distance_from_origin_vs_time_v7_256.pdf).
"""

from __future__ import annotations

import numpy as np

import _common

RUN_KEYS = ["fig1_m1", "fig1_m1p5", "single256_3000", "fig1_m3", "fig1_m4", "fig1_m5"]
PALETTE = ["#8051AD", "#476DB4", "#009DB9", "#3F9B74", "#DE9036", "#CE4856"]
STYLES = ["-", (0, (7, 2.5)), "-", (0, (5, 2, 1.4, 2)), (0, (3.2, 2)), (0, (1.2, 1.8))]


def compute(ctx, legacy_time_axis=False):
    series = []
    for key in RUN_KEYS:
        run = ctx.run(key, legacy_time_axis=legacy_time_axis)
        series.append({
            "key": key,
            "mass_percent": ctx.run_entry(key).get("mass_percent"),
            "time_Myr": run.times_myr,
            "radius_pc": run.distance_from_origin_pc(0),
        })
    return series


def plot(series, args):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": args.font, "font.size": 14, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "path.simplify": False, "axes.linewidth": 0.9,
                         "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(9.8, 6.1))
    fig.subplots_adjust(left=0.103, right=0.966, bottom=0.123, top=0.965)
    ax.set_xlim(args.xlim)
    ax.set_ylim(args.ylim)
    ax.set_xticks(np.arange(args.xlim[0], args.xlim[1] + 1, 500))
    ax.set_yticks(np.arange(args.ylim[0], args.ylim[1] + 1, 30))
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4, width=0.8)
    for item, color, style in zip(series, PALETTE, STYLES):
        ax.plot(item["time_Myr"], item["radius_pc"], color=color, linestyle=style, linewidth=1.25,
                dash_capstyle="round", solid_capstyle="round", label="%g%%" % item["mass_percent"])
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Distance from Origin (pc)")
    if args.legend:
        ax.legend(frameon=False)
    return fig


def main(argv=None):
    parser = _common.make_parser("fig01", __doc__.splitlines()[1])
    parser.add_argument("--legacy-time-axis", action="store_true",
                        help="use the old k*T/(save_number+1) axis (the published Fig. 1 did NOT)")
    parser.add_argument("--xlim", type=float, nargs=2, default=(0, 3000))
    parser.add_argument("--ylim", type=float, nargs=2, default=(0, 150))
    parser.add_argument("--font", default="Arial", help="font family (published: Arial)")
    parser.add_argument("--legend", action="store_true", help="add a legend (the published figure has none)")
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args)
    series = compute(ctx, args.legacy_time_axis)
    fig = plot(series, args)
    paths = ctx.save(fig, {}, png_dpi=210)  # refine_figure1.py: fig.savefig(path, dpi=210)
    ctx.write_provenance({"observable": "|x_BH| from simulation origin [pc]",
                          "legend_order": [s["mass_percent"] for s in series]})
    for p in paths:
        print(p)
    return series


if __name__ == "__main__":
    main()
