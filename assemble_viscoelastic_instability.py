#!/usr/bin/env python3
"""
Assemble Well Viscoelastic Instability dataset files into a single NetCDF file.
Based on the dataset configuration: 512x512 resolution, varying timesteps (20-60).
Fields: pressure, c_zz (scalars), velocity_x, velocity_y (vector), C_xx/xy/yx/yy (tensor components)
Constant fields: Re, Wi, beta, epsilon, Lmax (ignored in assembler)
"""

import argparse
import os
import glob
from netCDF4 import Dataset
import numpy as np


def assemble_viscoelastic_instability_data(input_dir, output_file):
    """Assemble all viscoelastic instability files into a single NetCDF file."""
    
    # Get all .nc files
    pattern = os.path.join(input_dir, "**", "*.hdf5")
    nc_files = glob.glob(pattern, recursive=True)
    nc_files.sort()
    
    print(f"Found {len(nc_files)} files to assemble")
    
    if not nc_files:
        raise ValueError(f"No .hdf5 files found in {input_dir}")
    
    # Analyze first file to get dimensions
    samples = [0]
    with Dataset(nc_files[0], "r") as first_nc:
        print(f"Variables in first file: {list(first_nc.variables.keys())}")
        print(f"Dimensions in first file: {list(first_nc.dimensions.keys())}")
        
        # Get dimensions
        n_trajectories = first_nc.dimensions["sample"].size
        n_timesteps = first_nc.dimensions["time"].size
        height = first_nc.dimensions["y"].size
        width = first_nc.dimensions["x"].size
        
        samples.append(n_trajectories)
        
        print(f"File structure: {n_trajectories} trajectories, {n_timesteps} timesteps, {height}x{width} resolution")
    
    # Count samples from all files
    for nc_file in nc_files[1:]:
        with Dataset(nc_file, "r") as nc:
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
        
        # Create variables for each field (ignoring constant fields Re, Wi, beta, epsilon, Lmax)
        print("Creating variables...")
        
        # Scalar fields: pressure, c_zz (sample, time, y, x)
        pressure_var = out_nc.createVariable(
            "pressure", "f4", ("sample", "time", "y", "x"),
            chunksizes=(1, 1, height, width)
        )
        
        c_zz_var = out_nc.createVariable(
            "c_zz", "f4", ("sample", "time", "y", "x"),
            chunksizes=(1, 1, height, width)
        )
        
        # Vector fields: velocity_x, velocity_y (sample, time, y, x, vector_dim)
        velocity_var = out_nc.createVariable(
            "velocity", "f4", ("sample", "time", "y", "x", "vector_dim"),
            chunksizes=(1, 1, height, width, 2)
        )
        
        # Tensor field: C_xx, C_xy, C_yx, C_yy -> flattened (sample, time, y, x, tensor_dim)
        c_tensor_var = out_nc.createVariable(
            "C", "f4", ("sample", "time", "y", "x", "tensor_dim"),
            chunksizes=(1, 1, height, width, 4)
        )
        
        # Copy data from all files
        for i, nc_file in enumerate(nc_files):
            print(f"Processing {nc_file} ({i+1}/{len(nc_files)})")
            
            with Dataset(nc_file, "r") as nc:
                # Read data for each field
                pressure_data = nc.variables['pressure'][:]
                c_zz_data = nc.variables['c_zz'][:]
                
                # Combine velocity components
                velocity_x_data = nc.variables['velocity_x'][:]
                velocity_y_data = nc.variables['velocity_y'][:]
                velocity_data = np.stack([velocity_x_data, velocity_y_data], axis=-1)
                
                # Combine C tensor components
                c_xx_data = nc.variables['C_xx'][:]
                c_xy_data = nc.variables['C_xy'][:]
                c_yx_data = nc.variables['C_yx'][:]
                c_yy_data = nc.variables['C_yy'][:]
                c_tensor_data = np.stack([c_xx_data, c_xy_data, c_yx_data, c_yy_data], axis=-1)
                
                # Write to output file
                start_idx = samples[i]
                end_idx = samples[i + 1]
                
                pressure_var[start_idx:end_idx] = pressure_data
                c_zz_var[start_idx:end_idx] = c_zz_data
                velocity_var[start_idx:end_idx] = velocity_data
                c_tensor_var[start_idx:end_idx] = c_tensor_data
        
        # Add metadata
        out_nc.setncattr("description", "Assembled Well Viscoelastic Instability dataset")
        out_nc.setncattr("total_samples", total_samples)
        out_nc.setncattr("timesteps", n_timesteps)
        out_nc.setncattr("spatial_resolution", f"{height}x{width}")
        out_nc.setncattr("fields", "pressure,c_zz,velocity,C")
        out_nc.setncattr("dataset_name", "viscoelastic_instability")
        out_nc.setncattr("boundary_conditions", "PERIODIC,WALL")
    
    print(f"Successfully assembled data to {output_file}")
    print(f"Final structure:")
    print(f"  - {total_samples} samples")
    print(f"  - {n_timesteps} timesteps")  
    print(f"  - {height}x{width} spatial resolution")
    print(f"  - 4 fields: pressure(1), c_zz(1), velocity(2), C(4) = 8 channels total")


def main():
    parser = argparse.ArgumentParser(description="Assemble Well Viscoelastic Instability dataset")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing viscoelastic instability .hdf5 files")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output assembled .nc file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    assemble_viscoelastic_instability_data(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
