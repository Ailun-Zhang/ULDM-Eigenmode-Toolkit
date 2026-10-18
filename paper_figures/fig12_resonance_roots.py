#!/usr/bin/env python3
"""Fig. 12 (fig:16): rescaled stationary roots mu_pm / kappa^2 of Eq. (C85) versus x = Gamma / kappa.

    mu_pm / kappa^2 = [2 - 4 x^2 +- sqrt(1 - 16 x^2 + 16 x^4)] / 3

Analytic; no data.  Roots are drawn only where they are real (discriminant >= 0), i.e.
x <= 1/sqrt(8 + 4 sqrt3) = 0.258819 and x >= 1/sqrt(8 - 4 sqrt3) = 0.965926; the two vertical lines
mark those thresholds.  The producing script of the published f_x.pdf (Matplotlib 3.7.5, 2025-12-14) was
not found; the style here was reconstructed from that PDF's vector content: 10.5 x 4.2 in, DejaVu Sans,
f_+ red / f_- blue (lw 2.4), threshold lines green / purple (lw 2.2), grid alpha 0.25, x in [0, 3],
y in [-2, 1.6], region labels "real / complex / real", legend lower right.
"""

from __future__ import annotations

import numpy as np

import _common

X_LOW = 1.0 / np.sqrt(8.0 + 4.0 * np.sqrt(3.0))   # 0.2588190...
X_HIGH = 1.0 / np.sqrt(8.0 - 4.0 * np.sqrt(3.0))  # 0.9659258...


def roots(x):
    """Return (mu_plus, mu_minus) / kappa^2; NaN where the discriminant is negative."""
    x = np.asarray(x, dtype=np.float64)
    disc = 1.0 - 16.0 * x ** 2 + 16.0 * x ** 4
    disc = np.where(np.abs(disc) < 1e-12, 0.0, disc)  # exact zero at the two thresholds
    root = np.sqrt(np.where(disc >= 0, disc, np.nan))
    return (2.0 - 4.0 * x ** 2 + root) / 3.0, (2.0 - 4.0 * x ** 2 - root) / 3.0


def x_grid(x_max, n):
    x = np.linspace(0.0, x_max, int(n))
    return np.unique(np.concatenate([x, [X_LOW, X_HIGH]]))


def plot(args):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    x = x_grid(args.xlim[1], args.n_points)
    f_plus, f_minus = roots(x)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(x, f_plus, color="red", lw=2.4, label=r"$f_+(x)$")
    ax.plot(x, f_minus, color="blue", lw=2.4, label=r"$f_-(x)$")
    ax.axvline(X_HIGH, color="green", lw=2.2,
               label=r"$x=\frac{1}{\sqrt{8-4\sqrt{3}}}\approx %.6f$" % X_HIGH)
    ax.axvline(X_LOW, color="purple", lw=2.2,
               label=r"$x=\frac{1}{\sqrt{8+4\sqrt{3}}}\approx %.6f$" % X_LOW)
    for xt, text in ((0.102, "real"), (0.612, "complex"), (2.658, "real")):
        ax.text(xt, 1.445, text, ha="center", va="center", fontsize=10)
    ax.set_xlim(*args.xlim)
    ax.set_ylim(*args.ylim)
    ax.set_xlabel(r"$x=\Gamma/\kappa$", fontsize=13)
    ax.set_ylabel(r"$f(x)$", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    return fig, x, f_plus, f_minus


def main(argv=None):
    parser = _common.make_parser("fig12", __doc__.splitlines()[1])
    parser.add_argument("--xlim", type=float, nargs=2, default=(0.0, 3.0))
    parser.add_argument("--ylim", type=float, nargs=2, default=(-2.0, 1.6))
    parser.add_argument("--n-points", type=int, default=6001)
    args = parser.parse_args(argv)
    ctx = _common.Context.from_args(args, needs_data=False)
    fig, x, f_plus, f_minus = plot(args)
    paths = ctx.save(fig, {})
    ctx.write_provenance({"equation": "C85", "x_low": float(X_LOW), "x_high": float(X_HIGH),
                          "checks": {"x0": [float(f_plus[0]), float(f_minus[0])]}})
    for p in paths:
        print(p)
    return x, f_plus, f_minus


if __name__ == "__main__":
    main()
