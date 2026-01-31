import sys
import os
import json
import numpy as np
import pyshtools as pysh
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
import multiprocessing as mp
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from Eig_ULDM_packages.functions3 import *
 
def process_r0_chunk(params):
    """Function to process a single r0 value"""
    file_index, r_0, l_max, grid_spacing_kpc, pool_size, file_prefix, num_points = params
    process_file(file_index, r_0, l_max, grid_spacing_kpc, pool_size, file_prefix, num_points)

def process_file_chunk(chunk_params):
    """Function to process file chunks, maintaining original logic"""
    start_idx, end_idx, r_0_values, l_max, grid_spacing_kpc, pool_size, file_prefix, num_points = chunk_params
    for r_0 in r_0_values:
        for file_index in range(start_idx, end_idx + 1):
            process_file(file_index, r_0, l_max, grid_spacing_kpc, pool_size // 14, file_prefix, num_points)

if __name__ == "__main__":
    # Parameter settings
    start_index = 500 # Starting index for processing 3WfnRS files
    end_index = 1000 # Ending index for processing 3WfnRS files
    r_0_values = np.load("r_0_values.npy") # Load r0 values from file
    l_max = 10 # Maximum degree for spherical harmonics
    resol = 256 # Grid resolution for 3WfnRS data
    length_kpc = 5 # Total length in kpc 
    grid_spacing_kpc = length_kpc / resol  # Convert parsec to kiloparsec
    pool_size = 56  # Total number of CPU cores
    cores_per_r0 = 4  # Number of cores allocated for each r0
    file_prefix = "3WfnRS/P3R_#" # or "3Wfn/P3D_#"
    num_points = 300  # Resolution for spherical harmonics coefficient calculation
    plot_num_points = 100  # Resolution for plotting
    duration = 3000  # Duration in Myr
    save_number = 501  # Number of saves

    # Check for special case (single time point, multiple r0 values)
    is_single_time = (start_index == end_index) and len(r_0_values) > 1

    # Define a function to process one time point with parallel r0 processing
    def process_time_point(file_index):
        print(f"\nProcessing time point {file_index}")
        
        # Calculate number of r0 values that can be processed simultaneously
        concurrent_r0s = pool_size // cores_per_r0
        print(f"Will process {concurrent_r0s} r0 values concurrently, each with {cores_per_r0} cores")

        # Split r0_values into batches
        r0_batches = [r_0_values[i:i + concurrent_r0s] 
                     for i in range(0, len(r_0_values), concurrent_r0s)]
        
        total_batches = len(r0_batches)
        print(f"Total number of batches: {total_batches}")

        # Process r0 values batch by batch
        for batch_idx, r0_batch in enumerate(r0_batches, 1):
            print(f"\nProcessing batch {batch_idx}/{total_batches}")
            
            # Create parameters for each r0 in current batch
            r0_params = [(file_index, r_0, l_max, grid_spacing_kpc, 
                         cores_per_r0, file_prefix, num_points) 
                        for r_0 in r0_batch]
            
            # Use process pool to parallel process r0 values in current batch
            with ProcessPoolExecutor(max_workers=concurrent_r0s) as executor:
                futures = [executor.submit(process_r0_chunk, params) 
                          for params in r0_params]
                
                # Show progress with tqdm
                for _ in tqdm(as_completed(futures), total=len(r0_batch), 
                             desc=f"Batch {batch_idx}/{total_batches}"):
                    pass

    if is_single_time:
        # Process the single time point
        process_time_point(start_index)
    else:
        # Process each time point sequentially, using the same logic as single time point
        for file_index in range(start_index, end_index + 1):
            process_time_point(file_index)

    # Create animation
    #create_animation(start_index, end_index, grid_spacing_kpc, r_0_values, l_max, 
                    #pool_size, file_prefix, plot_num_points, duration, save_number)