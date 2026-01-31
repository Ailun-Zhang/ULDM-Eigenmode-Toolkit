import os
import gc
import sys
import time
import json
import warnings
import numpy as np
import pyshtools as pysh
import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="paramiko")

def load_wave_function(filename):
    """Load the wave function data from a numpy file."""
    return np.load(filename)

def calculate_mass_center(wave_function, grid_spacing_kpc):
    """Calculate the mass center using the probability density (|ψ|²)."""
    density_data = np.abs(wave_function) ** 2  # Calculate probability density
    n = density_data.shape[0]
    indices = np.arange(n)
    x = y = z = indices * grid_spacing_kpc
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    total_mass = np.sum(density_data)
    mass_center_x = np.sum(X * density_data) / total_mass
    mass_center_y = np.sum(Y * density_data) / total_mass
    mass_center_z = np.sum(Z * density_data) / total_mass

    return np.array([mass_center_x, mass_center_y, mass_center_z])

def compute_wave_function_on_sphere_mapcoord(wf, mass_center, grid_spacing_kpc, r, theta, phi):
    # 球面点的 Cartesian 坐标
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    # 转成数组索引
    xi = (x + mass_center[0]) / grid_spacing_kpc
    yi = (y + mass_center[1]) / grid_spacing_kpc
    zi = (z + mass_center[2]) / grid_spacing_kpc
    coords = np.vstack([xi.ravel(), yi.ravel(), zi.ravel()])
    # 分别对实部和虚部做三次插值
    real_vals = map_coordinates(wf.real, coords, order=3, mode='constant', cval=0.0)
    imag_vals = map_coordinates(wf.imag, coords, order=3, mode='constant', cval=0.0)
    return (real_vals + 1j*imag_vals).reshape(theta.shape)

def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to Cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def compute_sh_coeffs(wave_function, mass_center, r, l_max, num_points, grid_spacing_kpc):
    """
    Compute the complex spherical harmonic coefficients using pyshtools.
    Vectorized implementation for better performance.
    
    Parameters remain the same for compatibility.
    """
    N = num_points
    theta = np.linspace(0, np.pi, N)
    phi = np.linspace(0, 2 * np.pi, 2 * N)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Vectorized computation of wave function values
    wave_function = compute_wave_function_on_sphere_mapcoord(
        wave_function, mass_center, grid_spacing_kpc,
        r, theta_grid, phi_grid
    )
    
    # Compute spherical harmonic coefficients
    coeffs = pysh.expand.SHExpandDHC(
        wave_function, norm=4, sampling=2, csphase=-1, lmax_calc=l_max
    )
    
    return coeffs

def load_existing_alm(filename):
    """Load existing complex alm coefficients from a JSON file if it exists."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            coeffs = np.array(data["coeffs"], dtype=np.complex128)
            return coeffs
    return None

def save_sh_coeffs(filename, source_file, r_0, coeffs):
    """Save the complex spherical harmonic coefficients to a JSON file."""
    data = {
        "source_file": source_file,
        "r_0": r_0,
        "coeffs": coeffs.tolist()  # NumPy complex arrays need special handling
    }
    with open(filename, 'w') as f:
        # Convert complex numbers to strings for JSON serialization
        json_str = json.dumps(data, default=lambda x: str(x) if isinstance(x, complex) else x)
        f.write(json_str)
 
def format_file_index(index):
    """
    Format file index to match file naming convention.
    Handles both 3-digit (000-999) and 4-digit (1000) formats.
    """
    if index < 1000:
        return f"{index:03d}"
    return f"{index:04d}"

def process_file(file_index, r_0, l_max, grid_spacing_kpc, pool_size, file_prefix, num_points):
    """Process a single wave function file with optimized interpolation timing."""
    source_file = f"{file_prefix}{format_file_index(file_index)}.npy"
    if os.path.exists(source_file):
        try:
            total_start_time = datetime.now()
            
            # Load wave function
            load_start_time = datetime.now()
            wave_function = load_wave_function(source_file)
            load_end_time = datetime.now()
            load_time = (load_end_time - load_start_time).total_seconds()
            
            # Calculate mass center
            center_start_time = datetime.now()
            mass_center = calculate_mass_center(wave_function, grid_spacing_kpc)
            center_end_time = datetime.now()
            center_time = (center_end_time - center_start_time).total_seconds()
            
            # Prepare spherical grid once per file
            theta = np.linspace(0, np.pi, num_points)
            phi   = np.linspace(0, 2 * np.pi, 2 * num_points)
            theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
            
            # Interpolation timing using map_coordinates
            interp_start_time = datetime.now()
            wf_on_sphere = compute_wave_function_on_sphere_mapcoord(
                wave_function, mass_center, grid_spacing_kpc,
                r_0, theta_grid, phi_grid
            )
            interp_end_time = datetime.now()
            interp_time = (interp_end_time - interp_start_time).total_seconds()
            
            # Spherical harmonic coefficient calculation timing
            calc_start_time = datetime.now()
            coeffs = pysh.expand.SHExpandDHC(
                wf_on_sphere, norm=4, sampling=2, csphase=-1, lmax_calc=l_max
            )
            calc_end_time = datetime.now()
            calc_time = (calc_end_time - calc_start_time).total_seconds()
            
            # Save coefficients
            source_file_name = os.path.splitext(os.path.basename(source_file))[0]
            output_dir = os.path.join(
                "alm_coefficients",
                file_prefix.split('/')[0],
                f"r0_{r_0}",
                f"lmax_{l_max}"
            )
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir,
                f"alm_coefficients_{source_file_name}_r0_{r_0}.json"
            )
            
            save_start_time = datetime.now()
            save_sh_coeffs(output_file, source_file, r_0, coeffs)
            save_end_time = datetime.now()
            save_time = (save_end_time - save_start_time).total_seconds()
            
            total_end_time = datetime.now()
            total_elapsed_time = (total_end_time - total_start_time).total_seconds()
            
            # Detailed logging
            with open("processed_files_detailed.txt", "a") as log_file:
                log_file.write(f"File: {source_file}, r_0={r_0}\n")
                log_file.write(f"  Total time: {total_elapsed_time:.2f} seconds\n")
                log_file.write(f"  Load time: {load_time:.2f} seconds\n")
                log_file.write(f"  Center calculation time: {center_time:.2f} seconds\n")
                log_file.write(f"  Interpolation time: {interp_time:.2f} seconds\n")
                log_file.write(f"  Coefficient calculation time: {calc_time:.2f} seconds\n")
                log_file.write(f"  Save time: {save_time:.2f} seconds\n")
                log_file.write("-" * 50 + "\n")
            
            # Simplified log
            with open("processed_files.txt", "a") as log_file:
                log_file.write(
                    f"Processed {source_file} for r_0={r_0} in {total_elapsed_time:.2f} seconds.\n"
                )
            
            del wave_function
            gc.collect()

        except Exception as e:
            print(f"Error processing file {source_file}: {e}")
            with open("error_log.txt", "a") as error_file:
                error_file.write(f"Error processing {source_file} for r_0={r_0}: {e}\n")
    else:
        print(f"File {source_file} does not exist.")
        with open("error_log.txt", "a") as error_file:
            error_file.write(f"File not found: {source_file}\n")

def load_alm_from_file(file_prefix, file_index, r_0, l_max):
    """Load the computed complex alm coefficients from a JSON file."""
    base_dir = "alm_coefficients"
    file_prefix_folder = file_prefix.split('/')[0]
    sub_dir = os.path.join(file_prefix_folder, f"r0_{r_0}", f"lmax_{l_max}")
    #filename = f"alm_coefficients_{os.path.basename(file_prefix)}{file_index:03d}_r0_{r_0}.json"
    filename = f"alm_coefficients_{os.path.basename(file_prefix)}{format_file_index(file_index)}_r0_{r_0}.json"
    file_path = os.path.join(base_dir, sub_dir, filename)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        r_0 = data["r_0"]
        # Convert string representations of complex numbers back to complex128
        coeffs_str = np.array(data["coeffs"])
        coeffs = np.zeros_like(coeffs_str, dtype=np.complex128)
        
        # Convert string representations back to complex numbers
        for i in range(coeffs_str.shape[0]):
            for j in range(coeffs_str.shape[1]):
                for k in range(coeffs_str.shape[2]):
                    complex_str = coeffs_str[i,j,k].strip('()')
                    if 'j' in complex_str:
                        coeffs[i,j,k] = complex(complex_str.replace(' ', ''))
                    else:
                        coeffs[i,j,k] = float(complex_str)
                        
    return r_0, coeffs

def batch_process_files(start_index, end_index, r_0_values, l_max, grid_spacing_kpc, pool_size, file_prefix, num_points):
    """Batch process files from start_index to end_index."""
    for r_0 in r_0_values:
        file_indices = list(range(start_index, end_index + 1))
        for file_index in file_indices:
            process_file(file_index, r_0, l_max, grid_spacing_kpc, pool_size, file_prefix, num_points)

def plot_interpolated_and_reconstructed_density(file_prefix, file_index, grid_spacing_kpc, r_0, plot_num_points, l_max, duration, save_number):
    """Create comparison plots of original and reconstructed probability densities using map_coordinates."""
    source_file = f"{file_prefix}{format_file_index(file_index)}.npy"
    wave_function = np.load(source_file)
    mass_center = calculate_mass_center(wave_function, grid_spacing_kpc)

    theta, phi = np.meshgrid(
        np.linspace(0, np.pi, plot_num_points),
        np.linspace(0, 2 * np.pi, plot_num_points),
        indexing='ij'
    )

    # Original probability density using map_coordinates interpolation
    wave_interpolated = compute_wave_function_on_sphere_mapcoord(
        wave_function, mass_center, grid_spacing_kpc, r_0, theta, phi
    )
    density_interpolated = np.abs(wave_interpolated) ** 2

    # Reconstructed wave function and density
    r_0, coeffs = load_alm_from_file(file_prefix, file_index, r_0, l_max)
    wave_reconstructed = np.zeros_like(theta, dtype=np.complex128)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            sph_harm = pysh.expand.spharm_lm(
                l, m, theta * 180 / np.pi, phi * 180 / np.pi,
                normalization='ortho', kind='complex', csphase=-1, degrees=True
            )
            if m >= 0:
                wave_reconstructed += coeffs[0, l, m] * sph_harm
            else:
                wave_reconstructed += coeffs[1, l, -m] * sph_harm

    density_reconstructed = np.abs(wave_reconstructed) ** 2

    # Create the plot
    time_myr = file_index * (duration / save_number)
    theta_mollweide = theta - np.pi / 2
    phi_mollweide = phi - np.pi

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'mollweide'}, figsize=(20, 10))
    fig.subplots_adjust(top=2.5, bottom=0.1, left=0.05, right=0.95, wspace=0.3)

    c1 = ax1.pcolormesh(phi_mollweide, theta_mollweide, density_interpolated, shading='auto', cmap='viridis')
    ax1.set_title(rf'Interpolated $|\psi(r_0={r_0} kpc, \theta,\phi)|^2$ at {time_myr:.2f} Myr', pad=20)
    ax1.grid(True)

    c2 = ax2.pcolormesh(phi_mollweide, theta_mollweide, density_reconstructed, shading='auto', cmap='viridis')
    ax2.set_title(rf'Reconstructed $|\psi(r_0={r_0} kpc, \theta,\phi)|^2$ from spherical harmonics ($l \leq {l_max}$) at {time_myr:.2f} Myr', pad=20)
    ax2.grid(True)

    # Set ticks and labels
    y_ticks = [-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2]
    y_labels = [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{3}$", r"$-\frac{\pi}{6}$", r"$0$", r"$\frac{\pi}{6}$", r"$\frac{\pi}{3}$", r"$\frac{\pi}{2}$"]
    x_ticks = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
    x_labels = [r"$-\pi$", r"$-\frac{2\pi}{3}$", r"$-\frac{\pi}{3}$", r"$0$", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$"]

    for ax in [ax1, ax2]:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

    cbar = fig.colorbar(c2, ax=[ax1, ax2], label='Density', orientation='horizontal', pad=0.05, fraction=0.05, aspect=30)
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    frame_filename = f"frame_{r_0}_{format_file_index(file_index)}.png"
    fig.savefig(frame_filename, dpi=200)
    plt.close(fig)
    gc.collect()

    return frame_filename

def combine_frames_to_video(frame_filenames, output_file, fps=5):
    """Combine multiple frames into a video."""
    clip = ImageSequenceClip(frame_filenames, fps=fps)
    clip.write_videofile(output_file, codec='libx264')
    for filename in frame_filenames:
        os.remove(filename)

def create_animation(start_index, end_index, grid_spacing_kpc, r_0_values, l_max, pool_size, file_prefix, plot_num_points, duration, save_number):
    """Create animation from processed wave function files."""
    output_dir = "density_animations_comparison"
    os.makedirs(output_dir, exist_ok=True)

    for r_0 in r_0_values:
        print(f"Processing r_0 = {r_0} kpc")
        start_time = time.time()
        
        with mp.Pool(pool_size) as pool:
            frame_filenames = pool.starmap(
                plot_interpolated_and_reconstructed_density, 
                [(file_prefix, i, grid_spacing_kpc, r_0, plot_num_points, l_max, duration, save_number) 
                 for i in range(start_index, end_index + 1)]
            )

        video_filename = os.path.join(output_dir, f"density_evolution_comparison_r0_{r_0:.2f}.mp4")
        combine_frames_to_video(frame_filenames, video_filename)

        end_time = time.time()
        print(f"Animation for r_0 = {r_0} kpc created and saved as '{video_filename}'. Time taken: {end_time - start_time} seconds.")