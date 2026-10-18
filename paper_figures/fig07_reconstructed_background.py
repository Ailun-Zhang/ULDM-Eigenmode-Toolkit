#!/usr/bin/env python3
"""Fig. 7 (fig:8a,b,c,f): BH distance from the ULDM centre of mass in reconstructed-background runs (N=128).

Panels (original notebook plot.ipynb, cell 5, MODE / COMPARE_N):
  compare_n0  (a): recon_nr3_nd0 (n_r=3, n_d=0, C0 solid) vs recon_nr0_nd0 (n_r=0, n_d=0, C1 dashed)
  compare_n1  (b): recon_nr3_nd1 vs recon_nr0_nd1
  compare_n2  (c): recon_nr3_nd2 vs recon_nr0_nd2
  collection2 (f): recon_nr0_nd0..3 (n_r=0), tab10 colours by n_d, dashed, two-column legend
Observable: |x_BH - x_ULDCOM| in pc (plot.ipynb ``load_run``).  Reduced model: DF A=0.75, BH responds to the
ULDM but does not source it.  Time axis: the published panels used k*3000/(500+1) Myr (last point at
2994.01 Myr, verified from the PDF vector paths), so ``--legacy-time-axis`` defaults to true.
Note: n_d=0 means the n=0, l=1 dipole IS present (IC Soliton_plus_n_0_ell_1 / ..._plus_n_0_ell_1).
"""

from __future__ import annotations

import _common

PANELS = {
    "compare_n0": ["recon_nr3_nd0", "recon_nr0_nd0"],
    "compare_n1": ["recon_nr3_nd1", "recon_nr0_nd1"],
    "compare_n2": ["recon_nr3_nd2", "recon_nr0_nd2"],
    "compare_n3": ["recon_nr3_nd3", "recon_nr0_nd3"],
    "collection1": ["recon_nr3_ndnone", "recon_nr3_nd0", "recon_nr3_nd1", "recon_nr3_nd2", "recon_nr3_nd3"],
    "collection2": ["recon_nr0_nd0", "recon_nr0_nd1", "recon_nr0_nd2", "recon_nr0_nd3"],
}
PUBLISHED_PANELS = ["compare_n0", "compare_n1", "compare_n2", "collection2"]


def run_meta(ctx, key):
    e = ctx.run_entry(key)
    nr, nd = e["nr"], e["nd"]
    return {"key": key, "extra_radial": nr > 0, "nr_max": nr, "nd_max": nd, "n_max": -1 if nd is None else nd}


def compute(ctx, panel, legacy_time_axis=True, cache=None):
    cache = {} if cache is None else cache
    series = []
    for key in PANELS[panel]:
        if (key, legacy_time_axis) not in cache:
            run = ctx.run(key, legacy_time_axis=legacy_time_axis)
            cache[(key, legacy_time_axis)] = {"time_Myr": run.times_myr,
                                              "distance_pc": run.distance_from_uldm_com_pc(0)}
        series.append((run_meta(ctx, key), cache[(key, legacy_time_axis)]))
    return series


def legend_fontsize(panel, args):
    """plot.ipynb cell 5 LEGEND_FONTSIZE: the published compare_n panels were exported with 14 pt legend
    text and collection2 with 10 pt (the value left in the saved notebook)."""
    if args.legend_fontsize is not None:
        return args.legend_fontsize
    return 14 if panel.startswith("compare_n") else 10


def plot(series, panel, args):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use("default")
    mpl.rcParams.update({
        "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": legend_fontsize(panel, args), "axes.linewidth": 0.9, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.grid": False, "mathtext.fontset": "stix", "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans L", "Liberation Sans", "Arial", "DejaVu Sans"],
        "ps.useafm": True, "ps.fonttype": 3, "axes.unicode_minus": False,
    })
    unique_ns = sorted({r["n_max"] for r, _ in series})
    cmap = plt.get_cmap("tab10")
    color_by_n = {n: cmap(i % 10) for i, n in enumerate(unique_ns)}
    linestyle_by_extra = {True: "-", False: "--"}
    color_by_compare = {True: "C0", False: "C1"}
    scale, ylabel = (1000.0, "Distance (pc)") if args.distance_unit == "pc" else (1.0, "Distance (kpc)")
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    for r, dat in series:
        nd_text = r"\mathrm{-}" if r["nd_max"] is None else str(r["nd_max"])
        label = rf"$n_{{r}}={r['nr_max']},\; n_{{d}}={nd_text}$"
        color = color_by_compare[r["extra_radial"]] if panel.startswith("compare_n") else color_by_n[r["n_max"]]
        ax.plot(dat["time_Myr"], dat["distance_pc"] / 1000.0 * scale, color=color,
                linestyle=linestyle_by_extra[r["extra_radial"]], linewidth=1.8, alpha=1.0, label=label)
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False, ncol=2 if len(series) > 3 else 1, handlelength=2.6)
    plt.tight_layout()
    return fig


def main(argv=None):
    parser = _common.make_parser("fig07", __doc__.splitlines()[1])
    parser.add_argument("--panels", nargs="+", default=PUBLISHED_PANELS, choices=sorted(PANELS))
    parser.add_argument("--legacy-time-axis", type=_common.str2bool, default=True,
                        help="old-notebook axis k*T/(save_number+1) (published: true)")
    parser.add_argument("--distance-unit", choices=("pc", "kpc"), default="pc")
    parser.add_argument("--legend-fontsize", type=float, default=None,
                        help="legend font size (default as published: 14 for compare_n*, 10 for collection*)")
    parser.add_argument("--name-template", default="stone_skipping_{panel}_pc_v2")
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args)
    cache, written = {}, []
    for panel in args.panels:
        series = compute(ctx, panel, args.legacy_time_axis, cache)
        fig = plot(series, panel, args)
        stem = args.name_template.format(panel=panel)
        written += ctx.save(fig, {"dpi": 300}, basename=stem)  # cell 5: savefig(format='eps', dpi=300)
    ctx.write_provenance({"observable": "|x_BH - x_ULDCOM| [pc]", "panels": args.panels},
                         basename="stone_skipping_panels")
    for p in written:
        print(p)
    return cache


if __name__ == "__main__":
    main()
