#!/usr/bin/env python3
"""Fig. 6 (fig:5, fig:6): |c_{1,1,-1}(t)|^2 and |c_{1,0,0}(t)|^2 / M_code for the single 2 % BH run (N=256).

Source: cnlm_single256_3000 (c_nlm_mass170.h5 of the single-BH run 20251016_185657@256),
the 501 saved samples (no resampling), 0-3000 Myr.  Normalisation |c|^2 / M_code, M_code = 12e6 Msun in
code units.  Plotting: the toolkit function cnlm_spectral_analysis.plot_cnlm_prd_single (packaged from the
original notebook 'compute_c_nlm_updated copy.ipynb', cell 12, which wrote the published PDFs).  The toolkit writes the EPS;
the other formats are saved from the same figure inside the same style context.
Colours: green for (1,1,-1) and red for (1,0,0), as published (the saved notebook cell hard-codes green;
the l=0 PDF was exported after editing it to red).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

import _common

DEFAULT_MODES = ["1,1,-1:green", "1,0,0:red"]


def parse_mode(text):
    key, _, color = text.partition(":")
    n, l, m = (int(v) for v in key.split(","))
    return (n, l, m), (color or "red")


def m_tag(m):
    return f"mneg{abs(m)}" if m < 0 else (f"mpos{m}" if m > 0 else "m0")


@contextlib.contextmanager
def capture_show(callback):
    """Run ``callback(fig)`` when the toolkit calls plt.show() (still inside its rc_context)."""
    import matplotlib.pyplot as plt

    original = plt.show

    def _show(*a, **k):
        callback(plt.gcf())

    plt.show = _show
    try:
        yield
    finally:
        plt.show = original


def main(argv=None):
    parser = _common.make_parser("fig06", __doc__.splitlines()[1])
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="n,l,m:colour entries")
    parser.add_argument("--xlim", type=float, nargs=2, default=(0.0, 3000.0))
    parser.add_argument("--figsize", type=float, nargs=2, default=(3.5, 2.6))
    parser.add_argument("--version-suffix", default="_v3", help="suffix of the published file names")
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args)

    from Eig_ULDM_packages.cnlm_spectral_analysis import plot_cnlm_prd_single
    import matplotlib.pyplot as plt

    t, c_all, _ = _common.load_cnlm(ctx.file("cnlm_single256_3000"))
    m_code = _common.soliton_mass_code(ctx, "cnlm_single256_3000")
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    written, peaks = [], {}
    for text in args.modes:
        key, color = parse_mode(text)
        n, l, m = key
        stem = args.basename or f"cnlm_n{n}_l{l}_{m_tag(m)}_original{args.version_suffix}"
        if args.basename and len(args.modes) > 1:
            stem = f"{args.basename}_n{n}_l{l}_{m_tag(m)}"
        paths = ctx.output_paths(stem)
        eps_path = paths.get("eps", ctx.out_dir / f"{stem}.eps")

        def save_other_formats(fig, paths=paths):
            for fmt, path in paths.items():
                if fmt == "eps":
                    continue
                kw = {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
                if fmt == "png":
                    kw["dpi"] = ctx.png_dpi_for()
                fig.savefig(path, format=fmt, **kw)

        with capture_show(save_other_formats):
            plot_cnlm_prd_single(t, c_all[key], key, normalization=m_code, save_path=eps_path,
                                 xlim=tuple(args.xlim), figsize=tuple(args.figsize), color=color)
        plt.close("all")
        if "eps" not in ctx.formats and eps_path.exists():
            eps_path.unlink()
        out = [p for fmt, p in paths.items()]
        written += out
        y = np.abs(c_all[key]) ** 2 / m_code
        peaks[str(key)] = {"max": float(y.max()), "t_at_max": float(t[int(np.argmax(y))])}
    ctx.written.extend(str(p) for p in written)
    ctx.write_provenance({"M_code": m_code, "series_max": peaks}, basename="cnlm_single_modes")
    for p in written:
        print(p)
    return peaks


if __name__ == "__main__":
    main()
