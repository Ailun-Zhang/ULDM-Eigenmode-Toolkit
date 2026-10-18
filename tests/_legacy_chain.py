"""The historical a_lm -> c_nlm chain, step by step as in Main.ipynb and compute_c_nlm.ipynb.

process_file per (snapshot, shell) -> JSON -> create_sh_hdf5 -> load_a_lm -> compute_cnlm -> save_cnlm_to_h5.
Runs in the current process (pyshtools' FFT is only bitwise reproducible within one process).
"""
import os
import shutil

import numpy as np


def run_legacy_chain(work, snap_dir, indices, dx_kpc, r_max_kpc, n_r, l_max, num_points, eigen_file,
                     final_time, cnlm_workers=1, axion_ev=1e-21):
    from Eig_ULDM_packages import functions3 as f3
    from Eig_ULDM_packages.alm_utils import create_sh_hdf5
    from Eig_ULDM_packages import units
    from Eig_ULDM_packages.c_nlm_integrator import load_eigenfunctions, load_a_lm, compute_cnlm, save_cnlm_to_h5

    units.set_axion_mass(axion_ev)
    work = str(work)
    os.makedirs(os.path.join(work, "3WfnRS"), exist_ok=True)
    for k in indices:
        shutil.copy(os.path.join(str(snap_dir), f"P3R_#{k:03d}.npy"), os.path.join(work, "3WfnRS"))
    cwd = os.getcwd()
    os.chdir(work)
    try:
        r0 = np.linspace(0, r_max_kpc, n_r, endpoint=True)
        r0[0] = 1e-5
        np.save("r_0_values.npy", r0)
        for k in indices:
            for r in np.load("r_0_values.npy"):
                f3.process_file(k, r, l_max, dx_kpc, 1, "3WfnRS/P3R_#", num_points)
        r_values = np.linspace(0, r_max_kpc, n_r, endpoint=True)
        r_values[0] = 1e-5
        create_sh_hdf5(output_path="PyUL_psi_alm.h5", start_index=indices[0], end_index=indices[-1], n_r=n_r,
                       l_max=l_max, num_points=num_points, r_values=r_values, file_prefix="3WfnRS/P3R_#",
                       n_workers=1)
        r_f, f_dict = load_eigenfunctions(eigen_file)
        r_max_eig = float(r_f[-1])
        rc = np.linspace(0.0, units.convert_between(r_max_kpc, 'kpc', '', 'l'), n_r, endpoint=True)
        rc[0] = units.convert_between(1e-5, 'kpc', '', 'l')
        r_al_grid = (rc / rc[-1]) * r_max_eig
        alm = load_a_lm("PyUL_psi_alm.h5", r_al=r_al_grid, final_time=final_time)
        t_ret, c_dict = compute_cnlm(r_f=r_f, f_dict=f_dict, r_al=alm['r'], t_al=alm['t'], a_lm=alm['a_lm'],
                                     r_interp_kind='cubic', time_interp=None, t_out=None, grid_choice='eigen',
                                     r_manual=None, num_workers=cnlm_workers, progress=False)
        save_cnlm_to_h5(c_dict, t_ret, "c_nlm.h5")
        return {"alm": os.path.join(work, "PyUL_psi_alm.h5"), "cnlm": os.path.join(work, "c_nlm.h5"),
                "r_al": r_al_grid, "t": t_ret, "c_dict": c_dict}
    finally:
        os.chdir(cwd)


def h5_datasets(path):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
    return out


def assert_h5_bitwise(p1, p2):
    d1, d2 = h5_datasets(p1), h5_datasets(p2)
    assert sorted(d1) == sorted(d2)
    for k in d1:
        a, b = np.asarray(d1[k]), np.asarray(d2[k])
        assert a.shape == b.shape and a.dtype == b.dtype, k
        assert np.array_equal(a, b), f"{k} differs: max abs {np.max(np.abs(a - b))}"
