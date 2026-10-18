#!/usr/bin/env python3
"""Fig. 3 (tex label fig:2): radial eigenfunctions f_{n,l}(r) for n = 0..4 (columns) and l = 0, 1, 2 (rows).

Source: the eigenbasis file eigs_mass_170_Eig_25_lmax_10_nig_5120_py2.h5 (config key eigs_basis,
sha256 2bb42868...), i.e. the basis used for every c_nlm file except the binary run.  r is in PyUL code
length units (0 .. 0.20605 = 2.5 kpc at 1e-21 eV), exactly as on the published axis.
Plotting: the toolkit function Eig_ULDM_packages.fnl_plotting.plot_eigenfunction_panel (the packaged
version of f_nl_python.ipynb cell 9, which produced the published f_nl_grid_3x5.eps) with its
defaults (<= 2000 plotted points per curve, 2.4 x 2.2 in per panel).  The toolkit writes the EPS
itself; the other formats are saved from the same figure object.
"""

from __future__ import annotations

import numpy as np

import _common


def load_basis(path, ells):
    import h5py

    eigs_data = {}
    with h5py.File(path, "r") as f:
        r_grid = np.asarray(f["ell/0/r"], dtype=np.float64)
        for ell in ells:
            grp = f[f"ell/{int(ell)}"]
            r_ell = np.asarray(grp["r"], dtype=np.float64)
            if not np.array_equal(r_ell, r_grid):
                raise ValueError(f"radial grid of l={ell} differs from l=0")
            eigs_data[f"ell/{int(ell)}"] = {"f": np.asarray(grp["f"], dtype=np.float64),
                                            "E": np.asarray(grp["E"], dtype=np.float64)}
    return r_grid, eigs_data


def main(argv=None):
    parser = _common.make_parser("fig03", __doc__.splitlines()[1])
    parser.add_argument("--ells", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ns", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-plot-points", type=int, default=2000)
    parser.add_argument("--width-per-col", type=float, default=2.4)
    parser.add_argument("--height-per-row", type=float, default=2.2)
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args)

    from Eig_ULDM_packages.fnl_plotting import plot_eigenfunction_panel

    r_grid, eigs_data = load_basis(ctx.file("eigs_basis"), args.ells)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    paths = ctx.output_paths()
    eps_path = paths.get("eps", ctx.out_dir / f"{ctx.basename}.eps")
    fig, _ = plot_eigenfunction_panel(
        r_grid, eigs_data, ells=args.ells, ns=args.ns, max_plot_points=args.max_plot_points,
        width_per_col=args.width_per_col, height_per_row=args.height_per_row,
        output_path=eps_path, show=False)
    others = [fmt for fmt in ctx.formats if fmt != "eps"]
    written = []
    for fmt in others:  # rcParams set by the toolkit (savefig.bbox tight, pad 0.02) still apply
        fig.savefig(paths[fmt], format=fmt, **({"dpi": ctx.png_dpi_for()} if fmt == "png" else {}))
        written.append(paths[fmt])
    if "eps" in ctx.formats:
        written.insert(0, eps_path)
    elif eps_path.exists():
        eps_path.unlink()  # EPS was not requested; the toolkit always writes one
    ctx.written.extend(str(p) for p in written)
    ctx.write_provenance({"n_grid": int(r_grid.size), "r_max_code": float(r_grid[-1])})
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
