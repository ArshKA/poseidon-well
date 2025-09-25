import netCDF4
import numpy as np
import argparse
from tqdm import tqdm

def add_trajectory_lengths(file_path):
    """
    Opens a NetCDF file, infers the length of each trajectory by finding the
    last non-zero timestep, and saves this information into a new
    'trajectory_lengths' variable.
    """
    try:
        dataset = netCDF4.Dataset(file_path, 'a') # Open in append mode
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")
        return

    if 'trajectory_lengths' in dataset.variables:
        print(f"'trajectory_lengths' variable already exists in {file_path}. No action taken.")
        dataset.close()
        return

    if 'sample' not in dataset.dimensions or 'pressure' not in dataset.variables:
        print("File does not appear to be a valid assembled dataset. Missing 'sample' dimension or 'pressure' variable.")
        dataset.close()
        return

    num_samples = dataset.dimensions['sample'].size
    all_lengths = []

    print(f"Inferring lengths for {num_samples} trajectories in {file_path}...")
    
    # Use 'pressure' as a representative variable to check for non-zero data
    pressure_var = dataset.variables['pressure']

    for i in tqdm(range(num_samples), desc="Processing trajectories"):
        # Load all time steps for one sample
        sample_data = pressure_var[i, :, :, :]
        
        # Find the last time step that is not all zeros
        non_zero_steps = np.where(np.any(sample_data, axis=(1, 2)))[0]
        
        if len(non_zero_steps) > 0:
            # The length is the last index + 1
            length = np.max(non_zero_steps) + 1
        else:
            length = 0 # This trajectory is empty
            
        all_lengths.append(length)

    print(f"Inferred lengths: min={min(all_lengths)}, max={max(all_lengths)}, avg={np.mean(all_lengths):.2f}")

    # Create and save the new variable
    print("Saving 'trajectory_lengths' variable to the file...")
    lengths_var = dataset.createVariable('trajectory_lengths', 'i4', ('sample',))
    lengths_var.description = "The number of time steps in each trajectory (inferred)."
    lengths_var[:] = all_lengths
    
    dataset.close()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add inferred trajectory lengths to an assembled NetCDF file.")
    parser.add_argument("netcdf_file", type=str, help="Path to the .nc file to patch.")
    args = parser.parse_args()
    
    add_trajectory_lengths(args.netcdf_file)