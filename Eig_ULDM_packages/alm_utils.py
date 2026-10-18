"""Utilities for generating and analyzing $a_{lm}$ HDF5 outputs."""

from __future__ import annotations

import json
import os
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from tqdm import tqdm

from ._compat import alm_json_path as _alm_json_path, mp_context as _mp_context, prefix_folder as _prefix_folder
from ._h5meta import write_attrs as _write_attrs, provenance as _provenance


def format_file_index(index: int) -> str:
    """
    Format file index to match file naming convention.
    Handles both 3-digit (000-999) and 4-digit (1000+) formats.

    Parameters:
    -----------
    index : int
        The file index number

    Returns:
    --------
    str
        Formatted string: 3 digits for indices < 1000, 4 digits otherwise
    """
    if index < 1000:
        return f"{index:03d}"
    return f"{index:04d}"


def reorganize_coeffs(coeffs: np.ndarray, l_max: int) -> np.ndarray:
    """
    Reorganize coefficients from the (2, l_max+1, l_max+1) format
    to a flattened array of size (l_max+1)^2.

    Parameters:
    -----------
    coeffs : ndarray
        Input coefficients array of shape (2, l_max+1, l_max+1)
    l_max : int
        Maximum angular momentum quantum number

    Returns:
    --------
    ndarray
        Reorganized coefficients of shape ((l_max+1)^2,)
    """
    n_lm = (l_max + 1) ** 2
    result = np.zeros(n_lm, dtype=np.complex128)

    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m >= 0:
                result[idx] = coeffs[0, l, m]
            else:
                result[idx] = coeffs[1, l, -m]
            idx += 1

    return result


def _process_time_point(args):
    """
    Process a single time point and return ylm data plus summary stats.
    """
    idx, source_idx, r_values, l_max, file_prefix, counts = args
    n_r = len(r_values)
    n_lm = (l_max + 1) ** 2
    ylm_data = np.zeros((n_r, n_lm), dtype=np.complex128)

    missing_count = 0
    error_count = 0
    missing_samples = []
    error_samples = []

    formatted_idx = format_file_index(source_idx)

    for r_idx, r_val in enumerate(r_values):
        json_path = _alm_json_path(
            file_prefix, r_val, l_max,
            f'alm_coefficients_{os.path.basename(file_prefix)}{formatted_idx}_r0_{r_val}.json',
        )

        try:
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                coeffs_str = np.array(data["coeffs"])

                # Convert string representations to complex numbers
                coeffs = np.zeros_like(coeffs_str, dtype=np.complex128)
                for i in range(coeffs_str.shape[0]):
                    for j in range(coeffs_str.shape[1]):
                        for k in range(coeffs_str.shape[2]):
                            complex_str = coeffs_str[i, j, k].strip('()')
                            if 'j' in complex_str:
                                coeffs[i, j, k] = complex(complex_str.replace(' ', ''))
                            else:
                                coeffs[i, j, k] = float(complex_str)

                # Convert alm to ylm and store in data array
                ylm = reorganize_coeffs(coeffs, l_max) * counts
                ylm_data[r_idx, :] = ylm

        except FileNotFoundError:
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(json_path)
        except Exception as exc:
            error_count += 1
            if len(error_samples) < 5:
                error_samples.append(f"{json_path} -> {exc}")

    return idx, source_idx, ylm_data, missing_count, error_count, missing_samples, error_samples


def create_sh_hdf5(
    output_path: str,
    start_index: int,
    end_index: int,
    n_r: int,
    l_max: int,
    num_points: int,
    r_values,
    file_prefix: str = "3WfnRS/P3R_#",
    mass_value: float = 170.3149822613627,
    n_workers: int | None = None,
    *,
    allow_missing: bool = False,
    times=None,
    time_units: str | None = None,
    r_units: str | None = None,
    angular_grid: str | None = None,
    metadata: dict | None = None,
):
    """
    Create an HDF5 file to store spherical harmonics expansion data.
    Enhanced version with consistent file path handling and flexible prefix support.

    Layout (unchanged since the paper): ``/massArr`` (n_times,) filled with
    ``mass_value``; one group ``alm_%06d`` per time (0-based position in
    ``start_index..end_index``) with datasets ``counts`` (n_r, (l_max+1)^2,
    constant ``num_points*2*num_points``) and ``ylm`` = a_lm * counts
    (complex128, columns l = 0..l_max, m = -l..l).

    Since version 0.3.0:

    * a missing or unreadable JSON file raises ``RuntimeError`` (listing the
      first failures) and no output file is left behind.  Pass
      ``allow_missing=True`` for the historical behaviour (zero rows); the
      file then records ``missing_count``, ``error_count``,
      ``missing_per_time`` and sample paths as root attributes.
    * provenance is written as root attributes only (r grid, snapshot
      indices, l_max, num_points, file prefix, software versions, and the
      optional ``times``/``time_units``/``r_units``/``angular_grid``/
      ``metadata``).  Attributes do not change what readers iterate over.
    * the file is written to ``<output_path>.tmp`` and renamed on success.

    Parameters:
    -----------
    output_path : str
        Path where the HDF5 file will be saved
    start_index : int
        Starting index for source files
    end_index : int
        Ending index for source files
    n_r : int
        Number of radial points
    l_max : int
        Maximum angular momentum quantum number
    num_points : int
        Number of points used in theta direction for original calculation
    r_values : array-like
        Array of r values used in the calculation
    file_prefix : str
        Prefix for source files (default: "3WfnRS/P3R_#")
    mass_value : float
        Value to fill the massArr dataset
    n_workers : int, optional
        Number of worker processes to use (defaults to cpu_count - 1)
    allow_missing : bool, optional
        Keep the historical zero-filling of missing/corrupt JSON files
        (default False: raise).
    times : array-like, optional
        Physical time of each snapshot (length end_index-start_index+1),
        stored as the root attribute ``times`` (read by ``load_a_lm``).
    time_units, r_units, angular_grid : str, optional
        Recorded as root attributes (``r_units`` is the unit of
        ``r_values``, e.g. 'kpc'; '' means code units).
    metadata : dict, optional
        Extra root attributes.
    """

    # Extract folder name from file prefix
    prefix_folder = _prefix_folder(file_prefix)
    r_values = np.asarray(r_values, dtype=np.float64)
    if r_values.ndim != 1 or r_values.size != int(n_r):
        raise ValueError(f"len(r_values)={r_values.size} does not match n_r={n_r}.")

    # Calculate derived parameters
    n_lm = (l_max + 1) ** 2  # Total number of (l,m) combinations
    counts = num_points * (2 * num_points)  # Total point count
    n_times = end_index - start_index + 1  # Number of time steps
    if n_times < 1:
        raise ValueError(f"end_index ({end_index}) must be >= start_index ({start_index}).")
    if times is not None:
        times = np.asarray(times, dtype=np.float64)
        if times.shape != (n_times,):
            raise ValueError(f"times must have shape ({n_times},), got {times.shape}.")

    available_cores = cpu_count()
    if n_workers is None:
        n_workers = max(1, available_cores - 1)
    else:
        n_workers = max(1, int(n_workers))
        if n_workers > available_cores:
            print(
                f"Requested {n_workers} workers, but only {available_cores} cores are available. "
                f"Using {available_cores}."
            )
            n_workers = available_cores

    print(f"Creating HDF5 file: {output_path}")
    print(f"Processing {n_times} time points with {n_r} radial points")
    print(f"Using l_max = {l_max}, resulting in {n_lm} (l,m) combinations")
    print(f"Using {n_workers} worker processes")

    tmp_path = str(output_path) + ".tmp"
    missing_per_time = np.zeros(n_times, dtype=np.int64)
    # Create HDF5 file and structure
    with h5py.File(tmp_path, 'w') as f:
        # Create mass array dataset
        mass_arr = np.full(n_times, mass_value, dtype=np.float64)
        f.create_dataset('massArr', data=mass_arr)

        # Pre-create counts dataset template
        counts_data = np.full((n_r, n_lm), counts, dtype=np.float64)

        # Prepare tasks for parallel processing
        tasks = [
            (idx, source_idx, r_values, l_max, file_prefix, counts)
            for idx, source_idx in enumerate(range(start_index, end_index + 1))
        ]
        chunksize = max(1, n_times // (n_workers * 4))

        total_missing = 0
        total_errors = 0
        missing_samples_all = []
        error_samples_all = []

        # Process time points in parallel and write results sequentially
        with _mp_context().Pool(processes=n_workers) as pool:
            results = pool.imap_unordered(_process_time_point, tasks, chunksize=chunksize)
            for result in tqdm(results, total=n_times, desc="Processing time points", unit="step"):
                idx, source_idx, ylm_data, missing_count, error_count, missing_samples, error_samples = result

                # Create group for this time point
                group_name = f'alm_{idx:06d}'
                group = f.create_group(group_name)
                group.create_dataset('counts', data=counts_data)
                group.create_dataset('ylm', data=ylm_data)

                total_missing += missing_count
                total_errors += error_count
                missing_per_time[idx] = missing_count + error_count
                for item in missing_samples:
                    if len(missing_samples_all) < 5:
                        missing_samples_all.append(item)
                for item in error_samples:
                    if len(error_samples_all) < 5:
                        error_samples_all.append(item)

        if total_missing > 0:
            print(f"Missing JSON files: {total_missing} (showing up to 5)")
            for path in missing_samples_all:
                print(f"  {path}")

        if total_errors > 0:
            print(f"Errors while parsing JSON files: {total_errors} (showing up to 5)")
            for item in error_samples_all:
                print(f"  {item}")

        meta = _provenance(
            "alm_utils.create_sh_hdf5",
            format="uldm-alm-h5",
            format_version=1,
            ylm_definition="ylm = a_lm * counts; columns l=0..l_max, m=-l..l",
            r_values=r_values,
            r_units="unknown" if r_units is None else str(r_units),
            n_r=int(n_r),
            l_max=int(l_max),
            num_points=int(num_points),
            counts=int(counts),
            file_prefix=str(file_prefix),
            start_index=int(start_index),
            end_index=int(end_index),
            snapshot_indices=np.arange(start_index, end_index + 1, dtype=np.int64),
            angular_grid="unknown (from JSON files)" if angular_grid is None else str(angular_grid),
            mass_value=float(mass_value),
            missing_count=int(total_missing),
            error_count=int(total_errors),
        )
        if times is not None:
            meta["times"] = times
            meta["time_units"] = "unknown" if time_units is None else str(time_units)
        if total_missing or total_errors:
            meta["missing_per_time"] = missing_per_time
            meta["missing_samples"] = list(missing_samples_all) + list(error_samples_all)
            meta["zero_filled"] = bool(allow_missing)
        if metadata:
            meta.update(metadata)
        _write_attrs(f.attrs, meta)

    if (total_missing or total_errors) and not allow_missing:
        os.remove(tmp_path)
        raise RuntimeError(
            f"create_sh_hdf5: {total_missing} missing and {total_errors} unreadable JSON files "
            f"(first: {(missing_samples_all + error_samples_all)[:3]}). No file was written. "
            "Fix the a_lm stage, or pass allow_missing=True to zero-fill them (historical behaviour)."
        )
    if total_missing or total_errors:
        print("WARNING: missing/unreadable shells were zero-filled (allow_missing=True); "
              "see the root attributes 'missing_per_time' and 'missing_samples'.")
    os.replace(tmp_path, output_path)
    print("HDF5 file creation complete.")


def analyze_h5_structure(filename: str):
    """Analyze and display the structure of an HDF5 file."""
    with h5py.File(filename, 'r') as f:
        print("=== HDF5 File Structure ===\n")

        def print_structure(name, obj):
            indent = '  ' * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}Dataset: {name}")
                print(f"{indent}`- Shape: {obj.shape}, Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")

        # Walk through the file structure
        f.visititems(print_structure)