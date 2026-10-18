"""Shared code of Figs. 9 and 10: seeded-dipole equal-mass binary vs the unseeded fully coupled binary.

The published Figs. 9 and 10 were drawn from curves digitised (PyMuPDF ``get_drawings``) out of earlier
figure PDFs; the digitiser scripts survive only in an archived working-session log (2026-06-24, not
deposited).  The scripts here
read the raw trajectories instead, with the observable those source PDFs plotted:
|x_BH| from the simulation origin in pc (Test_PyUL_alan_Binary.ipynb cells 25/27, ``distance_from_origin``).

* "unseeded": BH1 of the N=256 binary256 run (the source PDF 0.2_Binary_pc_v2.pdf drew one curve).
* "seeded A"/"seeded B": the two BHs of the N=128 seeded run.  Which body is A depends on the figure
  (it follows the drawing order of the digitised source PDF) and is a per-figure default.

All three source PDFs used the old-notebook time axis k*3000/(500+1) Myr, so the digitised curves end at
2994 Myr; ``--legacy-time-axis`` therefore defaults to true.  Style: the digitiser's ``plot()``
(7.0 x 2.72 in, Times New Roman 8.5 pt, black / grey 0.30 dashed / grey 0.62 dash-dot, y 0-215 pc).
"""

from __future__ import annotations

import _common

STYLES = {
    "unseeded": {"label": "unseeded", "color": "black", "linestyle": "-", "linewidth": 1.9, "zorder": 3},
    "seeded_A": {"label": "seeded A", "color": "0.30", "linestyle": (0, (6.0, 3.0)), "linewidth": 1.65, "zorder": 2},
    "seeded_B": {"label": "seeded B", "color": "0.62", "linestyle": (0, (7.0, 2.7, 1.7, 2.7)), "linewidth": 1.65,
                 "zorder": 1},
}


def add_arguments(parser, seeded_a_body, seeded_b_body):
    parser.add_argument("--legacy-time-axis", type=_common.str2bool, default=True,
                        help="old-notebook axis k*T/(save_number+1), as in the digitised source PDFs")
    parser.add_argument("--unseeded-body", type=int, default=0, help="body index of the unseeded curve")
    parser.add_argument("--seeded-a-body", type=int, default=seeded_a_body)
    parser.add_argument("--seeded-b-body", type=int, default=seeded_b_body)
    parser.add_argument("--observable", choices=("origin", "uldcom"), default="origin",
                        help="distance from the simulation origin (published source PDFs) or from the ULDM COM")
    parser.add_argument("--xlim", type=float, nargs=2, default=(0, 3000))
    parser.add_argument("--ylim", type=float, nargs=2, default=(0, 215))
    parser.add_argument("--font", default="Times New Roman")


def compute(ctx, seeded_key, args):
    def dist(run, body):
        return run.distance_from_origin_pc(body) if args.observable == "origin" else run.distance_from_uldm_com_pc(body)

    unseeded = ctx.run("binary256", legacy_time_axis=args.legacy_time_axis)
    seeded = ctx.run(seeded_key, legacy_time_axis=args.legacy_time_axis)
    return {
        "unseeded": (unseeded.times_myr, dist(unseeded, args.unseeded_body)),
        "seeded_A": (seeded.times_myr, dist(seeded, args.seeded_a_body)),
        "seeded_B": (seeded.times_myr, dist(seeded, args.seeded_b_body)),
    }


def plot(curves, args):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif", "font.serif": [args.font, "DejaVu Serif", "Times"],
        "font.size": 8.5, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "axes.linewidth": 0.8,
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.0, 2.72))
    for name in ("unseeded", "seeded_A", "seeded_B"):
        t, r = curves[name]
        ax.plot(t, r, **STYLES[name])
    ax.set_xlim(*args.xlim)
    ax.set_ylim(*args.ylim)
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Distance from origin (pc)" if args.observable == "origin" else "Distance from ULDM COM (pc)")
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.tick_params(direction="in", top=True, right=True, length=3.2, width=0.7)
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, 0.985), ncol=3, frameon=False, handlelength=2.8,
              handletextpad=0.45, columnspacing=1.6, borderaxespad=0.0)
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.19, top=0.965)
    return fig


def run_figure(fig_key, seeded_key, seeded_a_body, seeded_b_body, doc, argv=None):
    parser = _common.make_parser(fig_key, doc.splitlines()[1])
    add_arguments(parser, seeded_a_body, seeded_b_body)
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args)
    curves = compute(ctx, seeded_key, args)
    fig = plot(curves, args)
    paths = ctx.save(fig, {}, png_dpi=220)  # digitiser: fig.savefig(PDF_OUT); fig.savefig(PNG_OUT, dpi=220)
    ctx.write_provenance({"observable": "|x_BH| from simulation origin [pc]" if args.observable == "origin"
                          else "|x_BH - x_ULDCOM| [pc]",
                          "bodies": {"unseeded": args.unseeded_body, "seeded_A": args.seeded_a_body,
                                     "seeded_B": args.seeded_b_body},
                          "note": "published version digitised from PDFs; this one reads the raw NBody output"})
    for p in paths:
        print(p)
    return curves
