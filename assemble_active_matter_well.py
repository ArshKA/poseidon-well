#!/usr/bin/env python3
"""
Assemble Well Active Matter dataset files into a single NetCDF file.
Based on the dataset configuration: 256x256 resolution, 81 timesteps.
Fields: concentration (scalar), velocity_x, velocity_y (vector), D_xx/xy/yx/yy, E_xx/xy/yx/yy (tensors)
Constant fields: L, zeta, alpha (ignored in assembler)
"""

import argparse
import os
import glob
from netCDF4 import Dataset
import numpy as np


def assemble_active_matter_data(input_dir, output_file):
    """Assemble all active matter files into a single NetCDF file."""
    
    # Get all .nc or .hdf5 files (active matter uses .hdf5 format)
    patterns = [
        os.path.join(input_dir, "**", "*.nc"),
        os.path.join(input_dir, "**", "*.hdf5")
    ]
    nc_files = []
    for pattern in patterns:
        nc_files.extend(glob.glob(pattern, recursive=True))
    nc_files.sort()
    
    print(f"Found {len(nc_files)} files to assemble")
    
    if not nc_files:
        raise ValueError(f"No .nc or .hdf5 files found in {input_dir}")
    
    # Analyze first file to get dimensions
    samples = [0]
    with Dataset(nc_files[0], "r") as first_nc:
        print(f"Variables in first file: {list(first_nc.variables.keys())}")
        print(f"Dimensions in first file: {list(first_nc.dimensions.keys())}")
        
        # For active matter, check if it has the grouped structure
        if 't0_fields' in first_nc.groups:
            # Use grouped structure (like existing active matter assembler)
            conc_var = first_nc.groups['t0_fields'].variables['concentration']
            n_trajectories = conc_var.shape[0]
            n_timesteps = conc_var.shape[1] 
            height = conc_var.shape[2]
            width = conc_var.shape[3]
            print("Using grouped HDF5 structure")
        else:
            # Use standard structure
            n_trajectories = first_nc.dimensions["sample"].size
            n_timesteps = first_nc.dimensions["time"].size
            height = first_nc.dimensions["y"].size
            width = first_nc.dimensions["x"].size
            print("Using standard NetCDF structure")
        
        samples.append(n_trajectories)
        
        print(f"File structure: {n_trajectories} trajectories, {n_timesteps} timesteps, {height}x{width} resolution")
    
    # Count samples from all files
    for nc_file in nc_files[1:]:
        with Dataset(nc_file, "r") as nc:
            if 't0_fields' in nc.groups:
                conc_var = nc.groups['t0_fields'].variables['concentration']
                samples.append(conc_var.shape[0])
            else:
                samples.append(nc.dimensions["sample"].size)
    
    total_samples = sum(samples)
    
    # Calculate cumulative sample indices
    for i in range(1, len(samples)):
        samples[i] += samples[i - 1]
    
    print(f"Total samples: {total_samples}")
    
    # Create output file
    with Dataset(output_file, "w") as out_nc:
        # Create dimensions
        out_nc.createDimension("sample", total_samples)
        out_nc.createDimension("time", n_timesteps)
        out_nc.createDimension("x", width)
        out_nc.createDimension("y", height)
        out_nc.createDimension("vector_dim", 2)  # For velocity
        out_nc.createDimension("tensor_dim", 4)  # For flattened 2x2 tensors
        
        # Create variables for each field (ignoring constant fields L, zeta, alpha)
        print("Creating variables...")
        
        # Concentration: (sample, time, y, x)
        concentration_var = out_nc.createVariable(
            "concentration", "f4", ("sample", "time", "y", "x"),
            chunksizes=(1, 1, height, width)
        )
        
        # Velocity: (sample, time, y, x, vector_dim)
        velocity_var = out_nc.createVariable(
            "velocity", "f4", ("sample", "time", "y", "x", "vector_dim"),
            chunksizes=(1, 1, height, width, 2)
        )
        
        # D tensor: (sample, time, y, x, tensor_dim) - flattened 2x2
        d_var = out_nc.createVariable(
            "D", "f4", ("sample", "time", "y", "x", "tensor_dim"),
            chunksizes=(1, 1, height, width, 4)
        )
        
        # E tensor: (sample, time, y, x, tensor_dim) - flattened 2x2  
        e_var = out_nc.createVariable(
            "E", "f4", ("sample", "time", "y", "x", "tensor_dim"),
            chunksizes=(1, 1, height, width, 4)
        )
        
        # Copy data from all files
        for i, nc_file in enumerate(nc_files):
            print(f"Processing {nc_file} ({i+1}/{len(nc_files)})")
            
            with Dataset(nc_file, "r") as nc:
                if 't0_fields' in nc.groups:
                    # Use grouped structure
                    conc_data = nc.groups['t0_fields'].variables['concentration'][:]
                    vel_data = nc.groups['t1_fields'].variables['velocity'][:]
                    d_data = nc.groups['t2_fields'].variables['D'][:]
                    e_data = nc.groups['t2_fields'].variables['E'][:]
                    
                    # Flatten tensor data: (traj, time, y, x, 2, 2) -> (traj, time, y, x, 4)
                    d_flat = d_data.reshape(d_data.shape[0], d_data.shape[1], d_data.shape[2], d_data.shape[3], 4)
                    e_flat = e_data.reshape(e_data.shape[0], e_data.shape[1], e_data.shape[2], e_data.shape[3], 4)
                else:
                    # Use standard structure - reconstruct tensors from components
                    conc_data = nc.variables['concentration'][:]
                    
                    # Combine velocity components
                    velocity_x_data = nc.variables['velocity_x'][:]
                    velocity_y_data = nc.variables['velocity_y'][:]
                    vel_data = np.stack([velocity_x_data, velocity_y_data], axis=-1)
                    
                    # Combine D tensor components
                    d_xx = nc.variables['D_xx'][:]
                    d_xy = nc.variables['D_xy'][:]
                    d_yx = nc.variables['D_yx'][:]
                    d_yy = nc.variables['D_yy'][:]
                    d_flat = np.stack([d_xx, d_xy, d_yx, d_yy], axis=-1)
                    
                    # Combine E tensor components
                    e_xx = nc.variables['E_xx'][:]
                    e_xy = nc.variables['E_xy'][:]
                    e_yx = nc.variables['E_yx'][:]
                    e_yy = nc.variables['E_yy'][:]
                    e_flat = np.stack([e_xx, e_xy, e_yx, e_yy], axis=-1)
                
                # Write to output file
                start_idx = samples[i]
                end_idx = samples[i + 1]
                
                concentration_var[start_idx:end_idx] = conc_data
                velocity_var[start_idx:end_idx] = vel_data  
                d_var[start_idx:end_idx] = d_flat
                e_var[start_idx:end_idx] = e_flat
        
        # Add metadata
        out_nc.setncattr("description", "Assembled Well Active Matter dataset")
        out_nc.setncattr("total_samples", total_samples)
        out_nc.setncattr("timesteps", n_timesteps)
        out_nc.setncattr("spatial_resolution", f"{height}x{width}")
        out_nc.setncattr("fields", "concentration,velocity,D,E")
        out_nc.setncattr("dataset_name", "active_matter")
        out_nc.setncattr("boundary_conditions", "PERIODIC")
    
    print(f"Successfully assembled data to {output_file}")
    print(f"Final structure:")
    print(f"  - {total_samples} samples")
    print(f"  - {n_timesteps} timesteps")  
    print(f"  - {height}x{width} spatial resolution")
    print(f"  - 4 fields: concentration(1), velocity(2), D(4), E(4) = 11 channels total")


def main():
    parser = argparse.ArgumentParser(description="Assemble Well Active Matter dataset")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing active matter .nc or .hdf5 files")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output assembled .nc file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    assemble_active_matter_data(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
