"""T2/T3/T15: wavefunction tools under spawn, Y_lm backends, spectral plotting parameters."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Eig_ULDM_packages import wavefunction_tools as wt  # noqa: E402
from Eig_ULDM_packages import cnlm_spectral_analysis as csa  # noqa: E402


def test_ylm_backends_agree():
    th = np.linspace(0.1, 3.0, 7)
    ph = np.linspace(0.2, 6.0, 7)
    for l, m in [(0, 0), (1, -1), (3, 2)]:
        assert np.allclose(wt._sph_harm_ylm_pyshtools(l, m, th, ph), wt._sph_harm_ylm_scipy(l, m, th, ph),
                           rtol=1e-10, atol=1e-12)


def test_build_and_mode_fractions_with_portable_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("PYUL_AXION_MASS_EV", "1e-21")
    from Eig_ULDM_packages import units
    units.set_axion_mass(1e-21)
    r = np.linspace(0, 1.0, 200)
    f0 = np.exp(-(r / 0.2) ** 2)
    f0 /= np.sqrt(np.trapz(f0 ** 2 * r ** 2, r) if hasattr(np, "trapz") else np.trapezoid(f0 ** 2 * r ** 2, r))
    eigs = {"ell/0": {"f": f0[:, None]}}
    res = wt.build_initial_wavefunction(
        eigs_data=eigs, r_grid=r, length=2.0, length_units="", resol=16, CellCenteredGrid=True,
        mS_code_unit=4.0, s_velocity_unit="", nlm_list=[(0, 0, 0)], epsilon_nlm_list=[1.0 + 0j],
        center_of_mass=[0, 0, 0], COM_speed=[0, 0, 0], Phase=0, start_time=0, init_basename="T",
        nprocs=2, run_root_dir=str(tmp_path))
    assert res["Psi"].shape == (16, 16, 16) and np.isfinite(res["M_tot"]) and res["M_tot"] > 0
    frac = wt.compute_mode_fractions(
        eigs_data=eigs, r_grid=r, length=2.0, length_units="", resol=16, CellCenteredGrid=True,
        nlm_list=[(0, 0, 0)], center_of_mass=[0, 0, 0], psi_deboost_path=res["psi_path"], nprocs_modes=2)
    assert 0.5 < frac["total_mass"] / res["M_tot"] < 1.5
    with pytest.raises(FileNotFoundError, match="not found"):
        wt.compute_mode_fractions(eigs_data=eigs, r_grid=r, length=2.0, length_units="", resol=16,
                                  CellCenteredGrid=True, nlm_list=[(0, 0, 0)],
                                  deb_psipath=str(tmp_path / "nope.npy"))


def test_multimode_legend_matches_widths():
    t = np.linspace(0, 1, 20)
    c = {(0, l, m): (1 + l + m * 0.1) * np.exp(1j * t) for l in range(2) for m in range(-l, l + 1)}
    fig = csa.plot_cnlm_multimode(t, c, 0, 0, 1)
    title = fig.axes[0].get_legend().get_title().get_text()
    assert "m=0>m_->m_+" in title                           # default widths 3.5 > 2.5 > 1.5
    plt.close(fig)
    fig = csa.plot_cnlm_multimode(t, c, 0, 0, 1, lw_pos=3.0, lw_neg=1.0, time_unit="Gyr")
    assert "m=0>m_+>m_-" in fig.axes[0].get_legend().get_title().get_text()
    assert fig.axes[0].get_xlabel() == "Time $t$ (Gyr)"
    plt.close(fig)


def test_prd_single_xlim_and_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    t = np.linspace(0, 30000, 50)
    c = np.exp(1j * t / 1000)
    p = csa.plot_cnlm_prd_single(t, c, (1, 1, -1), save_path=tmp_path / "a.eps", xlim=None)
    assert os.path.exists(p)
    fig = plt.gcf()
    assert fig.axes[0].get_xlim() == (0.0, 30000.0) and fig.axes[0].get_xlabel() == "Time $t$ (Myr)"
    plt.close("all")


def test_print_spectral_summary_on_a_cp1252_console():
    """R9-11: the summary prints only ASCII (a redirected cp1252 stdout on Windows raised UnicodeEncodeError)."""
    import subprocess
    import sys
    from conftest import REPO

    code = ("import numpy as np\n"
            "from Eig_ULDM_packages.cnlm_spectral_analysis import analyze_cnlm_spectrum, print_spectral_summary\n"
            "t = np.linspace(0, 3000, 501)\n"
            "c = np.exp(2j*np.pi*0.01*t) + 0.6*np.exp(2j*np.pi*0.013*t)\n"
            "for key, sig in (((1, 1, -1), c), ((1, 1, 0), np.exp(2j*np.pi*0.01*t))):\n"
            "    print_spectral_summary(analyze_cnlm_spectrum(t, sig, key, peak_rel_threshold=0.01))\n")
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONPATH=str(REPO), MPLBACKEND="Agg")
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, env=env)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    text = out.stdout.decode("cp1252")
    assert "CONCLUSION" in text and "~=" in text
