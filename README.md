# ULDM Eigenmode Toolkit

A Python library for computing eigenfunctions $f_{nl}(r)$ and spectral coefficients $c_{nlm}(t)$ in ultralight dark matter (ULDM) simulations. This toolkit enables eigenmode decomposition and spectral analysis of ULDM wavefunctions from [PyUL_SK](https://github.com/Ailun-Zhang/PyUL_SK) simulation outputs.

## Overview

Ultralight dark matter (ULDM), also known as fuzzy dark matter, exhibits wave-like behavior on astrophysical scales. The wavefunction $\psi(\mathbf{r}, t)$ in a soliton potential can be expanded in eigenmodes:

$$\psi(\mathbf{r}, t) = \sum_{n,l,m} c_{nlm}(t)  f_{nl}(r)  Y_l^m(\theta, \phi)$$

where:
- $f_{nl}(r)$ are radial eigenfunctions solving the Schrödinger–Poisson equation
- $Y_l^m(\theta, \phi)$ are spherical harmonics
- $c_{nlm}(t)$ are time-dependent spectral coefficients encoding mode dynamics

This toolkit provides a complete pipeline for:
1. Computing radial eigenfunctions from soliton density profiles
2. Extracting spherical harmonic coefficients $a_{lm}(r, t)$ from 3D simulation data
3. Computing spectral coefficients $c_{nlm}(t)$ via radial integration
4. Performing time-domain and frequency-domain spectral analysis

---

## Features

- **Eigenfunction Solver**: Solve the radial eigenvalue problem for ULDM in a soliton potential
- **Spherical Harmonic Decomposition**: Extract $a_{lm}(r, t)$ from 3D wavefunction snapshots
- **Spectral Coefficient Computation**: Compute $c_{nlm}(t)$ from $a_{lm}$ and $f_{nl}$
- **Wavefunction Reconstruction**: Build initial conditions from eigenmode superpositions
- **Spectral Analysis**: FFT-based frequency analysis with automated peak detection
- **Publication-Ready Visualization**: Generate multi-panel diagnostic figures

---

## Installation

### Prerequisites

- Python 3.9 or later
- A working installation of [pyshtools](https://shtools.github.io/SHTOOLS/)

### Clone the Repository

```bash
git clone https://github.com/Ailun-Zhang/ULDM-Eigenmode-Toolkit.git
cd ULDM-Eigenmode-Toolkit
```

### Editable Installation (Recommended for Development)

Install the package in editable mode so that local changes take effect immediately:

```bash
pip install -e .
```

This installs `eig-uldm-packages` and all required dependencies:
- `numpy>=1.20`
- `scipy>=1.7`
- `matplotlib>=3.4`
- `h5py>=3.0`
- `tqdm>=4.60`
- `pyshtools>=4.10`

### Optional Development Dependencies

For running notebooks and creating animations:

```bash
pip install -e ".[dev]"
```

### Verify Installation

```python
from Eig_ULDM_packages.units import convert_between
from Eig_ULDM_packages.uldm_eig1 import make_all_eigs
from Eig_ULDM_packages.c_nlm_integrator import compute_cnlm
print("Installation successful!")
```

---

## Uninstallation

To remove the package:

```bash
pip uninstall eig-uldm-packages
```

---

## Quick Start

### Recommended Workflow

Start by reading **[Main.ipynb](Main.ipynb)**, which provides a linear entry point through the entire analysis pipeline:

```
Main.ipynb
    │
    ├── Step 1-3: Generate a_lm(r,t) from PyUL simulation data
    │
    ├── Step 4 → f_nl_python.ipynb (Part I)
    │             Compute eigenfunctions f_nl(r)
    │
    └── Step 5 → compute_c_nlm.ipynb
                  Compute c_nlm(t) and perform spectral analysis
```

---

## Notebooks

| Notebook | Purpose | Key Functions |
|----------|---------|---------------|
| **[Main.ipynb](Main.ipynb)** | Entry point; generates $a_{lm}(r,t)$ from simulation snapshots | `create_sh_hdf5()` |
| **[f_nl_python.ipynb](f_nl_python.ipynb)** | Solves eigenvalue problem; reconstructs wavefunctions | `make_all_eigs()`, `build_initial_wavefunction()` |
| **[compute_c_nlm.ipynb](compute_c_nlm.ipynb)** | Computes $c_{nlm}(t)$; spectral analysis and visualization | `compute_cnlm()`, `analyze_cnlm_spectrum()` |

---

## Package Structure

```
ULDM-Eigenmode-Toolkit/
├── Eig_ULDM_packages/          # Main Python package
│   ├── __init__.py
│   ├── units.py                # Unit conversion utilities
│   ├── uldm_eig1.py            # Eigenvalue solver
│   ├── alm_utils.py            # Spherical harmonic utilities
│   ├── c_nlm_integrator.py     # c_nlm computation engine
│   ├── cnlm_spectral_analysis.py  # Spectral analysis & plotting
│   ├── cnlm_postprocess.py     # Post-processing utilities
│   ├── wavefunction_tools.py   # Wavefunction construction
│   ├── fnl_plotting.py         # Eigenfunction visualization
│   ├── functions3.py           # Helper functions
│   └── soliton.h5              # Reference soliton profile
│
├── Main.ipynb                  # Entry point notebook
├── f_nl_python.ipynb           # Eigenfunction notebook
├── compute_c_nlm.ipynb         # Spectral analysis notebook
├── compute_coeff.py            # CLI script for a_lm extraction
├── submit_all_jobs_example.sh  # Example SLURM job script
│
├── pyproject.toml              # Package configuration
├── LICENSE                     # MIT License
├── .gitignore
└── README.md
```

---

## Module Reference

### `Eig_ULDM_packages.units`

Unit conversion between dimensionless simulation units and physical units (SI, kpc, Myr, etc.).

```python
from Eig_ULDM_packages.units import convert, convert_back, convert_between

# Convert 100 kpc to dimensionless units
r_code = convert(100, 'kpc', 'l')

# Convert back to physical units
r_physical = convert_back(r_code, 'kpc', 'l')
```

### `Eig_ULDM_packages.uldm_eig1`

Solve the radial eigenvalue problem.

```python
from Eig_ULDM_packages.uldm_eig1 import make_all_eigs, save_eigs_to_h5

# Compute eigenfunctions for l=0,1,2
eig_dict = make_all_eigs(ell_list=[0, 1, 2], n_per_ell=5, ...)

# Save to HDF5
save_eigs_to_h5(eig_dict, "eigenfunctions.h5")
```

### `Eig_ULDM_packages.c_nlm_integrator`

Compute spectral coefficients $c_{nlm}(t)$.

```python
from Eig_ULDM_packages.c_nlm_integrator import load_eigenfunctions, load_a_lm, compute_cnlm

# Load data
eig_data = load_eigenfunctions("eigenfunctions.h5")
a_lm_data = load_a_lm("a_lm.h5")

# Compute c_nlm
c_nlm_dict = compute_cnlm(eig_data, a_lm_data)
```

### `Eig_ULDM_packages.cnlm_spectral_analysis`

Spectral analysis and visualization.

```python
from Eig_ULDM_packages.cnlm_spectral_analysis import analyze_cnlm_spectrum, plot_4panel_eda

# Analyze spectrum
results = analyze_cnlm_spectrum(c_nlm_dict, dt=0.1, ...)

# Generate 4-panel diagnostic figure
fig = plot_4panel_eda(results, ...)
```

### `Eig_ULDM_packages.wavefunction_tools`

Construct and manipulate wavefunctions.

```python
from Eig_ULDM_packages.wavefunction_tools import (
    prepare_epsilon_nlm,
    build_initial_wavefunction,
    deboost_wavefunction,
    compute_mode_fractions
)

# Build initial wavefunction from mode amplitudes
result = build_initial_wavefunction(modes, epsilons, radial_fn, ...)

# Deboost to target COM and velocity
deboost_wavefunction(result['psi_path'], ...)
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYUL_AXION_MASS_EV` | Axion mass in eV | `1e-22` |

Set before importing modules:
```python
import os
os.environ["PYUL_AXION_MASS_EV"] = "1e-21"
from Eig_ULDM_packages.units import *
```

---

## Example Workflow

### 1. Generate Eigenfunctions

```python
# In f_nl_python.ipynb
from Eig_ULDM_packages.uldm_eig1 import *

# Load soliton profile
sol = load_soliton_file("Eig_ULDM_packages/soliton.h5")

# Compute eigenfunctions
eig_dict = make_all_eigs(
    sol['rho'], sol['r_grid'],
    ell_list=[0, 1, 2],
    n_per_ell=5,
    M_sol=1e9  # Solar masses
)

# Save results
save_eigs_to_h5(eig_dict, "eigenfunctions.h5")
```

### 2. Extract Spherical Harmonics from Simulation

```python
# In Main.ipynb
from Eig_ULDM_packages.alm_utils import create_sh_hdf5

create_sh_hdf5(
    sim_dir="/path/to/simulation",
    output_file="a_lm.h5",
    l_max=4
)
```

### 3. Compute and Analyze Spectral Coefficients

```python
# In compute_c_nlm.ipynb
from Eig_ULDM_packages.c_nlm_integrator import *
from Eig_ULDM_packages.cnlm_spectral_analysis import *

# Load data
eig = load_eigenfunctions("eigenfunctions.h5")
alm = load_a_lm("a_lm.h5")

# Compute c_nlm
c_nlm = compute_cnlm(eig, alm)

# Spectral analysis
results = analyze_cnlm_spectrum(c_nlm, dt=0.1, theoretical_freqs=eig['eigenvalues'])

# Visualize
fig = plot_4panel_eda(results, title="Mode Analysis")
fig.savefig("spectral_analysis.png", dpi=150)
```

---

## Related Projects

- **[PyUL_SK](https://github.com/Ailun-Zhang/PyUL_SK)**: ULDM simulation framework that generates the input data for this toolkit
- **[pyshtools](https://shtools.github.io/SHTOOLS/)**: Spherical harmonic transforms used in this toolkit

---

## Acknowledgments

Parts of this toolkit, including the eigenfunction solver methodology and the reference soliton profile, were inspired by Mathematica code generously provided by [J. Luna Zagorac](https://orcid.org/0000-0003-4504-1677). The theoretical foundation for the eigenmode decomposition approach is based on:

> Zagorac, J. L., Sands, I., Padmanabhan, N., & Easther, R. (2022). *Schrödinger-Poisson solitons: Perturbation theory*. Physical Review D, **105**, 103506.  
> DOI: [10.1103/PhysRevD.105.103506](https://doi.org/10.1103/PhysRevD.105.103506)

---

## Citation

If you use this toolkit in your research, please cite the following:

### Software

```bibtex
@software{uldm_eigenmode_toolkit,
  author = {Zhang, Alan},
  title = {ULDM Eigenmode Toolkit},
  year = {2026},
  url = {https://github.com/Ailun-Zhang/ULDM-Eigenmode-Toolkit}
}
```

### Associated Paper

```bibtex
@article{zhang2026stoneskipping,
  author = {Zhang, Alan and Wang, Yourong and Zagorac, J. Luna and Easther, Richard},
  title = {Stone Skipping Black Holes in Ultralight Dark Matter Solitons},
  journal = {arXiv preprint},
  year = {2026},
  month = {February},
  note = {[arXiv link](https://arxiv.org/abs/2602.11512)}
}
```

### Theoretical Foundation

```bibtex
@article{zagorac2022schrodinger,
  author = {Zagorac, J. L. and Sands, I. and Padmanabhan, N. and Easther, R.},
  title = {Schrödinger-Poisson solitons: Perturbation theory},
  journal = {Physical Review D},
  volume = {105},
  pages = {103506},
  year = {2022},
  doi = {10.1103/PhysRevD.105.103506}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## Contact

For questions or collaboration inquiries, please open an issue on GitHub or contact the author.
