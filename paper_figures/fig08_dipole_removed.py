#!/usr/bin/env python3
"""Fig. 8 (fig:dipole_removed): BH distance from the ULDM centre of mass, (a) all n<=2, l<=2 modes at a
common amplitude, (b) the same with every l=1 mode removed (reduced model, N=128, DF A=0.75).

Source runs: byhand_all (run ID 20251026_121920@128, all n<=2, l<=2 modes "by hand") and
byhand_no_l1 (run ID 20251029_033519@128, the same without l=1).
Observable: 3-D distance |x_BH - x_ULDCOM| in pc (original notebook Test_PyUL_alan_2.ipynb, cell 31, ``r3d``); ``--quantity``
also offers |z| and R_perp (the notebook's other two exports).  Time axis: the published figure used
k*3000/(500+1) Myr with x-limits [t_0, t_last] = [0, 2994.01] (verified from the PDF), so
``--legacy-time-axis`` defaults to true.  Style: cell 31 (6.9 x 2.7 in, Okabe-Ito blue/vermillion).
"""

from __future__ import annotations

import numpy as np

import _common


def compute(ctx, legacy_time_axis=True):
    out = []
    for key in ("byhand_all", "byhand_no_l1"):
        run = ctx.run(key, legacy_time_axis=legacy_time_axis)
        rel = run.relative_position_pc(0, frame="uldcom")
        out.append({"key": key, "t": run.times_myr, "r3d": np.sqrt(np.sum(rel ** 2, axis=1)),
                    "absz": np.abs(rel[:, 2]), "rxy": np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)})
    return out


def plot(data, args):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use("default")
    rc = {
        "figure.dpi": 120, "savefig.dpi": 300, "text.usetex": False, "mathtext.fontset": "cm",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans L", "Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12, "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 5.0, "ytick.major.size": 5.0, "xtick.minor.size": 2.8, "ytick.minor.size": 2.8,
        "legend.frameon": False, "legend.fontsize": 10.5, "ps.useafm": True, "ps.fonttype": 3,
        "ps.usedistiller": None, "pdf.use14corefonts": True, "pdf.fonttype": 42,
    }
    for k, v in rc.items():
        if k in mpl.rcParams:
            mpl.rcParams[k] = v
    unit = args.distance_unit
    scale = 1.0 if unit == "pc" else 1e-3
    ylabel = {"r3d": r"$r\;(\mathrm{%s})$", "absz": r"$|z|\;(\mathrm{%s})$",
              "rxy": r"$R_{\perp}\;(\mathrm{%s})$"}[args.quantity] % unit
    left, right = data
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7), sharey=True, constrained_layout=True)
    axL, axR = axes
    for ax in axes:
        ax.minorticks_on()
        ax.grid(False)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.set_facecolor("white")
    colours = ("#0072B2", "#D55E00") if args.colour else ("black", "black")
    axL.plot(left["t"], left[args.quantity] * scale, color=colours[0], lw=1.55)
    axR.plot(right["t"], right[args.quantity] * scale, color=colours[1], lw=1.55)
    xlabel = r"$\mathrm{Time\;(Myr)}$"
    axL.set_xlabel(xlabel)
    axR.set_xlabel(xlabel)
    axL.set_ylabel(ylabel)
    axL.text(0.02, 0.95, "(a)", transform=axL.transAxes, ha="left", va="top")
    axR.text(0.02, 0.95, "(b)", transform=axR.transAxes, ha="left", va="top")
    for ax, d in ((axL, left), (axR, right)):
        if len(d["t"]) > 1:
            ax.set_xlim(float(d["t"][0]), float(d["t"][-1]))
    return fig


def main(argv=None):
    parser = _common.make_parser("fig08", __doc__.splitlines()[1])
    parser.add_argument("--legacy-time-axis", type=_common.str2bool, default=True,
                        help="old-notebook axis k*T/(save_number+1) (published: true)")
    parser.add_argument("--quantity", choices=("r3d", "absz", "rxy"), default="r3d")
    parser.add_argument("--distance-unit", choices=("pc", "kpc"), default="pc")
    parser.add_argument("--colour", type=_common.str2bool, default=True, help="USE_COLOR_LINES (published: true)")
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args)
    data = compute(ctx, args.legacy_time_axis)
    fig = plot(data, args)
    paths = ctx.save(fig, {"bbox_inches": "tight", "facecolor": "white", "transparent": False})
    ctx.write_provenance({"observable": f"{args.quantity} of x_BH - x_ULDCOM [{args.distance_unit}]"})
    for p in paths:
        print(p)
    return data


if __name__ == "__main__":
    main()
