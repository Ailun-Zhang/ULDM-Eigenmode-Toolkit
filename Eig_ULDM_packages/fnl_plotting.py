"""Plotting utilities for the f_nl workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors


def _lighten(color, amount: float = 0.65):
    """Blend a color with white by `amount` (0->original, 1->white)."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - amount) * rgb + amount * np.ones(3))


def _nice_bounds(vmin: float, vmax: float, include_zero: bool = True):
    """Round bounds to 'nice' values for prettier axes."""
    if include_zero:
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.0)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return (vmin - 1.0, vmax + 1.0)
    span = vmax - vmin
    pad = 0.06 * span
    vmin -= pad
    vmax += pad
    rough_step = (vmax - vmin) / 4.0
    if rough_step <= 0:
        return (vmin, vmax)
    mag = 10 ** np.floor(np.log10(rough_step))
    norm = rough_step / mag
    if norm <= 1:
        step = 1 * mag
    elif norm <= 2:
        step = 2 * mag
    elif norm <= 5:
        step = 5 * mag
    else:
        step = 10 * mag
    vmin = step * np.floor(vmin / step)
    vmax = step * np.ceil(vmax / step)
    return (float(vmin), float(vmax))


def _make_edge_hiding_formatter(xmin: float, xmax: float, hide_left: bool = False, hide_right: bool = False):
    def _fmt(x, pos):
        if hide_left and np.isclose(x, xmin):
            return ""
        if hide_right and np.isclose(x, xmax):
            return ""
        return f"{x:g}"

    return mticker.FuncFormatter(_fmt)


def plot_eigenfunction_panel(
    r_grid,
    eigs_data,
    *,
    ells=(0, 1, 2),
    ns=(0, 1, 2, 3, 4),
    max_plot_points: int = 2000,
    width_per_col: float = 2.4,
    height_per_row: float = 2.2,
    ell_color_overrides: dict | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Render the paper-ready eigenfunction panel and save an EPS file."""
    if r_grid is None or eigs_data is None:
        raise NameError("Missing required inputs: r_grid and eigs_data.")

    r = np.asarray(r_grid)
    ells = list(ells)
    ns = list(ns)

    _cycle = plt.rcParams.get("axes.prop_cycle")
    _cycle_colors = _cycle.by_key().get(
        "color",
        ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    )
    ell_color = {ell: _cycle_colors[i % len(_cycle_colors)] for i, ell in enumerate(ells)}
    ell_color.update({0: "tab:red", 1: "tab:green", 2: "tab:purple"})
    if ell_color_overrides:
        ell_color.update(ell_color_overrides)

    if len(ells) == 0 or len(ns) == 0:
        raise ValueError("`ells` and `ns` must be non-empty lists.")
    if any((not isinstance(ell, (int, np.integer))) for ell in ells):
        raise TypeError("All entries in `ells` must be integers.")
    if any((not isinstance(n, (int, np.integer))) for n in ns):
        raise TypeError("All entries in `ns` must be integers.")
    if any(n < 0 for n in ns):
        raise ValueError("All entries in `ns` must be >= 0.")

    for ell in ells:
        key = f"ell/{int(ell)}"
        if key not in eigs_data:
            raise KeyError(
                f"eigs_data does not contain '{key}'. Available keys include: {list(eigs_data.keys())[:8]} ..."
            )
        f_mat = np.asarray(eigs_data[key]["f"])
        if f_mat.ndim != 2:
            raise ValueError(
                f"eigs_data['{key}']['f'] must be a 2D array, got shape {getattr(f_mat, 'shape', None)}"
            )
        max_n = int(np.max(ns))
        if max_n >= f_mat.shape[1]:
            raise IndexError(
                f"For ell={ell}, requested max n={max_n} but only {f_mat.shape[1]} eigenfunctions are available (n=0..{f_mat.shape[1]-1})."
            )

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
        }
    )

    nrows = len(ells)
    ncols = len(ns)
    figsize = (width_per_col * ncols, height_per_row * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=False,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    xmin = float(np.nanmin(r))
    xmax = float(np.nanmax(r))

    row_ylims = {}
    for ell in ells:
        f_mat = np.asarray(eigs_data[f"ell/{int(ell)}"]["f"])
        row_y = np.concatenate([f_mat[:, int(n)].reshape(-1) for n in ns])
        row_ylims[int(ell)] = _nice_bounds(
            float(np.nanmin(row_y)), float(np.nanmax(row_y)), include_zero=True
        )

    for i, ell in enumerate(ells):
        ell_i = int(ell)
        f_mat = np.asarray(eigs_data[f"ell/{ell_i}"]["f"])
        ylo, yhi = row_ylims[ell_i]
        for j, n in enumerate(ns):
            n_j = int(n)
            ax = axes[i, j]
            y = f_mat[:, n_j]

            if r.size > max_plot_points:
                step = int(np.ceil(r.size / max_plot_points))
                r_plot = r[::step]
                y_plot = y[::step]
            else:
                step = 1
                r_plot = r
                y_plot = y

            base_color = ell_color[ell_i]
            fill_color = _lighten(base_color, amount=0.72)

            ax.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0, zorder=1)
            ax.fill_between(
                r_plot,
                0.0,
                y_plot,
                where=(y_plot >= 0),
                interpolate=True,
                color=fill_color,
                linewidth=0.0,
                zorder=1.5,
            )
            ax.fill_between(
                r_plot,
                0.0,
                y_plot,
                where=(y_plot < 0),
                interpolate=True,
                color=fill_color,
                linewidth=0.0,
                zorder=1.5,
            )
            ax.plot(r_plot, y_plot, color=base_color, linewidth=1.6, zorder=2.5)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ylo, yhi)
            ax.grid(False, which="both")
            ax.set_title("")

            if j == 0:
                ax.set_ylabel(r"$f_{n,\ell}(r)$")
                ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            if i == nrows - 1:
                ax.set_xlabel("r")
                hide_left = (j != 0)
                hide_right = (j != ncols - 1)
                ax.xaxis.set_major_formatter(
                    _make_edge_hiding_formatter(xmin, xmax, hide_left=hide_left, hide_right=hide_right)
                )
                ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if i == 0:
                ax.annotate(
                    rf"$n={n_j}$",
                    xy=(0.5, 1.07),
                    xycoords="axes fraction",
                    ha="center",
                    va="bottom",
                )

            if j == ncols - 1:
                ax.annotate(
                    rf"$\ell={ell_i}$",
                    xy=(1.06, 0.5),
                    xycoords="axes fraction",
                    ha="left",
                    va="center",
                    rotation=90,
                )

    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.10, top=0.90, wspace=0.0, hspace=0.0)

    if output_path is None:
        out_path = Path(f"f_nl_grid_{nrows}x{ncols}.eps")
    else:
        out_path = Path(output_path)
    fig.savefig(out_path, format="eps")
    print(f"Saved EPS to: {out_path.resolve()}")
    step = int(np.ceil(r.size / max_plot_points)) if r.size > max_plot_points else 1
    print(
        f"Downsampling: r has {r.size} points, plotted with <= {max_plot_points} (step={step})"
    )

    if show:
        plt.show()

    return fig, axes