"""EQUIVALENCE on real PyUL output (skipped when the run is not available).

Set ULDM_EQUIV_RUN_DIR to a PyUL run directory that contains Outputs/3WfnRS/P3R_#000..003.npy
(the WP2 check used a 64^3, 5 kpc, 10-save run of the untouched PyUL_SK 30.40 code).  The
historical chain and the new pipeline are run in the same process, so the comparison is bitwise.
"""
import glob
import os

import numpy as np
import pytest

from _legacy_chain import assert_h5_bitwise, run_legacy_chain
from Eig_ULDM_packages import pipeline
from Eig_ULDM_packages.pyul_run import PyULRun

def _find_run():
    """A real PyUL run directory with Outputs/3WfnRS/P3R_#000..003.npy, given by ULDM_EQUIV_RUN_DIR.

    ULDM_EQUIV_RUN_DIR may point at the run directory itself or at a save path holding one '*@<N>' run."""
    cand = os.environ.get("ULDM_EQUIV_RUN_DIR")
    if not cand:
        return None
    if os.path.isdir(os.path.join(cand, "Outputs")):
        return cand
    hits = sorted(glob.glob(os.path.join(cand, "*@*")))
    return hits[0] if hits else None


RUN = _find_run()


@pytest.mark.skipif(RUN is None or not os.path.exists(os.path.join(str(RUN), "Outputs", "3WfnRS", "P3R_#003.npy")),
                    reason="real PyUL baseline run not available (set ULDM_EQUIV_RUN_DIR)")
def test_real_run_bitwise(tmp_path, small_eigen_file):
    run = PyULRun.open(RUN)
    n = np.load(os.path.join(run.outputs_dir, "3WfnRS", "P3R_#000.npy"), mmap_mode="r").shape[0]
    side = 5.0  # passed consistently to both chains (the run does not record it)
    out = tmp_path / "pipe"
    res = pipeline.run_pipeline(pipeline.build_parser().parse_args(
        [RUN, "--rs-side", str(side), "--rs-resol", str(n), "--eigen-file", small_eigen_file, "--snapshots", "0:3",
         "--n-shells", "17", "--num-points", "64", "--workers", "4", "--out-dir", str(out), "--alm-out", "alm.h5"]))
    final_time = float(run.snapshot_times_myr([3])[0])
    leg = run_legacy_chain(tmp_path / "legacy", os.path.join(run.outputs_dir, "3WfnRS"), [0, 1, 2, 3], side / n,
                           side / 2, 17, 2, 64, small_eigen_file, final_time=final_time)
    assert_h5_bitwise(leg["alm"], res["alm_file"])
    assert_h5_bitwise(leg["cnlm"], res["cnlm_file"])
