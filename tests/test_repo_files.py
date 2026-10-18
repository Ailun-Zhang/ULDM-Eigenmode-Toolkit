"""T12 compute_coeff.py, notebooks (template fixes, no stored outputs), packaging metadata, shell templates."""
import json
import os
import re
import subprocess
import sys

import nbformat
import numpy as np
import pytest

from conftest import REPO, write_run

NEEDLE = 'r_0_values = np.load("r_0_values.npy") # Load r0 values from file'


def test_compute_coeff_template_and_run(tmp_path):
    src = (REPO / "compute_coeff.py").read_text(encoding="utf-8")
    assert src.count(NEEDLE) == 1                        # Main.ipynb Step 1 substitutes this line
    assert "pool_size = 56" not in src and 'if __name__ == "__main__":' in src
    run = write_run(tmp_path, n=16, n_snap=3)
    outputs = run / "Outputs"
    r = np.linspace(0, 2.5, 4)
    r[0] = 1e-5
    np.save(outputs / "r_0_values.npy", r)
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-B", str(REPO / "compute_coeff.py"), "--num-points", "16", "--l-max", "2",
           "--workers", "2"]
    p = subprocess.run(cmd, cwd=outputs, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "snapshots 0..2" in p.stdout and "reproducibility.uldm" in p.stdout   # paper defaults + run metadata
    js = list((outputs / "alm_coefficients" / "3WfnRS").rglob("alm_coefficients_*.json"))
    assert len(js) == 3 * 4
    # a missing snapshot is reported with a non-zero exit code
    os.remove(outputs / "3WfnRS" / "P3R_#002.npy")
    p = subprocess.run(cmd + ["--rs-length-kpc", "5.0"], cwd=outputs, env=env, capture_output=True, text=True)
    assert p.returncode == 1 and "FAILED" in p.stderr


def test_compute_coeff_skip_existing_checks_kernel_params(tmp_path):
    """R9-02: --skip-existing refuses to reuse JSON files made with other kernel parameters."""
    run = write_run(tmp_path, n=16, n_snap=2)
    outputs = run / "Outputs"
    r = np.array([1e-5, 1.0, 2.0])
    np.save(outputs / "r_0_values.npy", r)
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    base = [sys.executable, "-B", str(REPO / "compute_coeff.py"), "--num-points", "16", "--l-max", "2",
            "--workers", "1", "--rs-length-kpc", "5.0"]

    def run_cc(*extra):
        return subprocess.run(base + list(extra), cwd=outputs, env=env, capture_output=True, text=True)

    p = run_cc()
    assert p.returncode == 0, p.stdout + p.stderr
    tree = outputs / "alm_coefficients" / "3WfnRS"
    sidecar = tree / "kernel_params_P3R_#lmax_2.json"
    rec = json.loads(sidecar.read_text())
    assert rec["grid_spacing_kpc"] == 5.0 / 16 and rec["angular_grid"] == "legacy" and rec["num_points"] == 16
    js = sorted(tree.rglob("alm_coefficients_*.json"))
    assert len(js) == 2 * 3 and not list(tree.rglob("*.tmp*"))
    before = {q: q.read_bytes() for q in js}
    # same parameters: everything is skipped
    p = run_cc("--skip-existing")
    assert p.returncode == 0 and "all 0 shells done" in p.stdout, p.stdout + p.stderr
    # other dx or angular grid: refused, nothing touched
    for extra in (["--rs-length-kpc", "4.0"], ["--angular-grid", "dh"], ["--num-points", "12"]):
        p = run_cc("--skip-existing", *extra)
        assert p.returncode != 0 and "different kernel parameters" in p.stderr, (extra, p.stdout + p.stderr)
        p = run_cc(*extra)                           # also without --skip-existing (would mix the tree)
        assert p.returncode != 0 and "different kernel parameters" in p.stderr, extra
    assert {q: q.read_bytes() for q in js} == before
    # a tree without sidecar (made before the check) is not trusted silently
    sidecar.unlink()
    p = run_cc("--skip-existing")
    assert p.returncode != 0 and "--no-params-check" in p.stderr
    p = run_cc("--skip-existing", "--no-params-check")
    assert p.returncode == 0 and "all 0 shells done" in p.stdout and sidecar.is_file()


def test_alm_json_legacy_absolute_prefix_location(tmp_path, monkeypatch):
    """R9-03: 0.2.0 trees written with an absolute POSIX prefix (alm_coefficients/r0_<r>/...) are still read."""
    from Eig_ULDM_packages import functions3 as f3
    from Eig_ULDM_packages._compat import alm_json_path

    monkeypatch.chdir(tmp_path)
    prefix = "/data/run/Outputs/3WfnRS/P3R_#"
    name = "alm_coefficients_P3R_#000_r0_0.5.json"
    new = os.path.join("alm_coefficients", "3WfnRS", "r0_0.5", "lmax_2", name)
    assert alm_json_path(prefix, 0.5, 2, name) == new                 # nothing exists: new location
    old_dir = tmp_path / "alm_coefficients" / "r0_0.5" / "lmax_2"     # 0.2.0: prefix.split('/')[0] == ''
    old_dir.mkdir(parents=True)
    coeffs = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3) * (1 + 0.5j)
    f3.save_sh_coeffs(str(old_dir / name), prefix + "000.npy", 0.5, coeffs)
    assert alm_json_path(prefix, 0.5, 2, name) == os.path.join("alm_coefficients", "", "r0_0.5", "lmax_2", name)
    r0, back = f3.load_alm_from_file(prefix, 0, 0.5, 2)
    assert r0 == 0.5 and np.array_equal(back, coeffs)
    assert alm_json_path("3WfnRS/P3R_#", 0.5, 2, name) == new        # relative prefixes: unchanged


NOTEBOOKS = ["Main.ipynb", "compute_c_nlm.ipynb", "f_nl_python.ipynb"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebooks_clean(name):
    raw = (REPO / name).read_bytes()
    assert b"\r\n" not in raw
    nb = nbformat.read(str(REPO / name), as_version=4)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == "code":
            assert c.outputs == [] and c.execution_count is None
    text = raw.decode("utf-8")
    # No absolute cluster paths (e.g. /data/<volume>/<user>/..., /scratch/<volume>/<user>/...) in shipped notebooks;
    # documentation placeholders such as /data/user/... are allowed.
    assert not re.search(r"/(?:data|scratch|home)/(?!user/)[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/", text)


def test_notebook_template_fixes():
    main = nbformat.read(str(REPO / "Main.ipynb"), as_version=4)
    src = "\n".join(c.source for c in main.cells if c.cell_type == "code")
    assert "start_index = 500" not in src and "start_index = 0" in src and "run.save_number" in src
    assert 'r_values = np.load("r_0_values.npy")' in src
    cnlm = nbformat.read(str(REPO / "compute_c_nlm.ipynb"), as_version=4)
    src = "\n".join(c.source for c in cnlm.cells if c.cell_type == "code")
    assert "rescale_r_to_eigen_rmax(r_values, r_max_eig)" in src
    assert src.index('os.environ["PYUL_AXION_MASS_EV"]') < src.index("import Eig_ULDM_packages")
    fnl = nbformat.read(str(REPO / "f_nl_python.ipynb"), as_version=4)
    src = "\n".join(c.source for c in fnl.cells if c.cell_type == "code")
    assert "excludes r=0" not in src and "INCLUDES r=0" in src
    assert src.index('os.environ["PYUL_AXION_MASS_EV"]') < src.index("from Eig_ULDM_packages")


def test_manifest_ships_what_the_tests_read():
    """R9-12: the sdist carries conftest.py, _legacy_chain.py and the repository files the tests read."""
    lines = {ln.strip() for ln in (REPO / "MANIFEST.in").read_text(encoding="utf-8").splitlines()}
    for needed in ("graft tests", "graft paper_figures", "graft examples"):
        assert needed in lines, needed
    text = " ".join(lines)
    for name in ("compute_coeff.py", "submit_all_jobs_example.sh", "Main.ipynb", "compute_c_nlm.ipynb",
                 "f_nl_python.ipynb", "CHANGELOG.md", ".zenodo.json"):
        assert name in text, name


def test_package_metadata():
    import Eig_ULDM_packages
    text = (REPO / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == Eig_ULDM_packages.__version__ == "0.3.0"
    assert 'uldm-cnlm = "Eig_ULDM_packages.pipeline:main"' in text
    assert "moviepy" not in text.split("[project.optional-dependencies]")[0]
    assert (REPO / "CHANGELOG.md").exists()


@pytest.mark.parametrize("path", ["examples/pbs/run_cnlm_pipeline.pbs", "examples/slurm/run_cnlm_pipeline.slurm",
                                  "submit_all_jobs_example.sh"])
def test_shell_templates(path):
    text = (REPO / path).read_text(encoding="utf-8")
    assert b"\r\n" not in (REPO / path).read_bytes()
    assert "set -euo pipefail" in text
    assert "a_lm.ipynb" not in text
    if path.endswith(".sh"):
        assert 'rm -f "${SCRIPT_DIR}/compute_coeff' not in text and 'rm -f "submit_job' not in text
        assert "NODE_LIST=()" in text                      # host pinning only if the user asks for it
    else:
        assert "host=" not in text                         # templates never pin hosts
        assert "%j" in text or "PBS_JOBID" in text         # unique logs


def _orcid_ok(orcid):
    digits = orcid.replace("-", "")
    total = 0
    for ch in digits[:-1]:
        total = (total + int(ch)) * 2
    check = (12 - total % 11) % 11
    return re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", orcid) and digits[-1] == ("X" if check == 10 else str(check))


def test_release_metadata():
    """R10: .zenodo.json is complete for the Zenodo GitHub integration; NOTICE, errata page and links exist."""
    raw = (REPO / ".zenodo.json").read_text(encoding="utf-8")
    assert "TODO" not in raw
    meta = json.loads(raw)
    assert meta["upload_type"] == "software" and meta["license"].lower() == "mit"  # Zenodo vocabulary id
    import Eig_ULDM_packages
    assert meta["version"] == Eig_ULDM_packages.__version__
    for creator in meta["creators"]:
        assert creator["name"] and creator["affiliation"] and _orcid_ok(creator["orcid"]), creator
    rel = {(r["identifier"], r["relation"]) for r in meta["related_identifiers"]}
    assert ("arXiv:2602.11512", "isSupplementTo") in rel
    assert ("https://github.com/Ailun-Zhang/PyUL_SK", "isSupplementTo") in rel
    for r in meta["related_identifiers"]:
        assert r["scheme"] in ("doi", "url", "arxiv"), r
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "TODO" not in readme and "arXiv:2602.11512" in readme
    for target in ("docs/conventions_and_errata.md", "NOTICE", "LICENSE", "CHANGELOG.md"):
        assert f"({target}" in readme and (REPO / target).is_file(), target
    assert "PyUltraLight" in (REPO / "NOTICE").read_text(encoding="utf-8")
    assert (REPO / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    assert '"NOTICE"' in (REPO / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
    assert "NOTICE" in manifest and "graft docs" in manifest
