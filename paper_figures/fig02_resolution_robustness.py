#!/usr/bin/env python3
"""Fig. 2 (tex label fig:resolution_robustness): (a) orbital radius r_sys and (b) dipole occupation P_{l=1}^{(n<=2)} for N=128/192/256.

Source runs: res128 (the N=128 rerun, run ID 20260920_131633@128, 30000 Myr / 500 saves), res192 (3000 Myr /
50 saves), res256 (30000 Myr / 500 saves); all saved every 60 Myr.  Window 0-3000 Myr (51 samples per run).
The published N=128 curves come from that rerun; ``--run-128 res128_jan2026`` uses the earlier
January-2026 N=128 run (run ID 20260118_004908@128, supplementary) instead.
(a) r_sys = |x_BH| from the simulation origin (initial total-COM frame), pc, from Outputs/NBody.
(b) P_dip(t) = sum_{n=0..2} sum_{m=-1..1} |c_{n1m}(t)|^2 / M_code with M_code = 12e6 Msun in code units
    (170.3149822613627 at 1e-21 eV), from each run's c_nlm HDF5 file (no time interpolation).
Style: the original figure script make_resolution_figure.py of the 2026-10-08 figure update, which drew the
published figure from derived/resolution_128_192_256.csv; ``--source csv`` replots that CSV instead of the raw data.
"""

from __future__ import annotations

import csv

import numpy as np

import _common

RUNS = ("128", "192", "256")
COLORS = {"128": "#0072B2", "192": "#D55E00", "256": "#222222"}
LINESTYLES = {"128": (0, (5, 2)), "192": (0, (5, 2, 1.5, 2)), "256": "-"}
MARKERS = {"128": "s", "192": "D", "256": "x"}
MARK_EVERY = {"128": (3, 8), "192": (5, 8), "256": (1, 8)}
DRAW_ORDER = ("256", "192", "128")


def dipole_occupation(t_c, c_dict, m_code, n_max=2, ell=1):
    """sum_{n<=n_max, m} |c_{n,ell,m}|^2 / m_code (raw samples)."""
    total = np.zeros(t_c.shape, dtype=np.float64)
    for n in range(n_max + 1):
        for m in range(-ell, ell + 1):
            total += np.abs(np.asarray(c_dict[(n, ell, m)])) ** 2
    return total / m_code


def compute_raw(ctx, t_max, run_keys=None):
    run_keys = run_keys or {res: "res" + res for res in RUNS}
    out = {}
    for res in RUNS:
        run = ctx.run(run_keys[res]).window(0.0, t_max)
        t = run.times_myr
        r_sys = run.distance_from_origin_pc(0)
        key = "cnlm_" + run_keys[res]
        t_c, c_dict, _ = _common.load_cnlm(ctx.file(key))
        sel = t_c <= t_max + 1e-9
        if sel.sum() != t.size or not np.allclose(t_c[sel], t, rtol=0, atol=1e-6):
            raise ValueError(f"N={res}: c_nlm times do not match the NBody snapshot times in 0-{t_max} Myr")
        m_code = _common.soliton_mass_code(ctx, key)
        pdip = dipole_occupation(t_c[sel], {k: v[sel] for k, v in c_dict.items() if k[1] == 1}, m_code)
        out[res] = {"t_myr": t, "r_sys_pc": r_sys, "Pdip_nle2": pdip, "M_code": m_code, "run": run_keys[res]}
    return out


def compute_csv(ctx, t_max):
    path = ctx.file("derived_resolution_csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    t = np.asarray([float(r["t_myr"]) for r in rows])
    sel = (t >= 0) & (t <= t_max)
    return {res: {"t_myr": t[sel],
                  "r_sys_pc": np.asarray([float(r[f"r_sys_pc_{res}"]) for r in rows])[sel],
                  "Pdip_nle2": np.asarray([float(r[f"Pdip_nle2_{res}"]) for r in rows])[sel]} for res in RUNS}


def plot(data, args):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.45,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.025,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.00, 2.48), gridspec_kw={"width_ratios": [1, 1]},
                             constrained_layout=True)
    for panel, (quantity, ylim, ylabel) in enumerate((
            ("r_sys_pc", args.ylim_a, r"orbital radius $r_{\rm sys}$ [pc]"),
            ("Pdip_nle2", args.ylim_b, r"$P_{\ell=1}^{(n\leq 2)}$"))):
        ax = axes[panel]
        for res in DRAW_ORDER:
            d = data[res]
            ax.plot(d["t_myr"], d[quantity], color=COLORS[res], linestyle=LINESTYLES[res], marker=MARKERS[res],
                    markevery=MARK_EVERY[res], markersize=3.0, markerfacecolor="white", markeredgewidth=0.85,
                    label=rf"$N={res}$", zorder=2)
        ax.set_xlim(0, args.t_max)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"time [Myr]")
        ax.set_ylabel(ylabel)
    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(0.025, 0.975, label, transform=axis.transAxes, ha="left", va="top", fontweight="bold")
        axis.grid(True, color="#E3E3E3", linewidth=0.45)
        axis.tick_params(direction="in", top=True, right=True, width=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               frameon=False, handlelength=3.1, handletextpad=0.55, columnspacing=2.2)
    engine = fig.get_layout_engine() if hasattr(fig, "get_layout_engine") else None
    if engine is not None:
        engine.set(rect=(0.0, 0.0, 1.0, 0.90))
    else:  # matplotlib < 3.6
        fig.set_constrained_layout_pads(rect=(0.0, 0.0, 1.0, 0.90))
    return fig


def main(argv=None):
    parser = _common.make_parser("fig02", __doc__.splitlines()[1])
    parser.add_argument("--source", choices=("raw", "csv"), default="raw",
                        help="raw NBody + c_nlm files (default) or the derived CSV used for the published figure")
    parser.add_argument("--run-128", default="res128",
                        help="config run key for N=128 (published: res128, the N=128 rerun; alternative: res128_jan2026)")
    parser.add_argument("--t-max", type=float, default=3000.0)
    parser.add_argument("--ylim-a", type=float, nargs=2, default=(0, 145))
    parser.add_argument("--ylim-b", type=float, nargs=2, default=(0, 0.0064))
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args)
    keys = {"128": args.run_128, "192": "res192", "256": "res256"}
    data = compute_raw(ctx, args.t_max, keys) if args.source == "raw" else compute_csv(ctx, args.t_max)
    fig = plot(data, args)
    paths = ctx.save(fig, {}, png_dpi=400)
    ctx.write_provenance({"observables": {"a": "|x_BH| from simulation origin [pc]",
                                          "b": "sum_{n<=2,m} |c_n1m|^2 / M_code"},
                          "M_code": {k: v.get("M_code") for k, v in data.items()}})
    for p in paths:
        print(p)
    return data


if __name__ == "__main__":
    main()
