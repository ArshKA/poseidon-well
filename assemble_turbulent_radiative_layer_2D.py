#!/usr/bin/env python3
"""
Assemble Well Turbulent Radiative Layer 2D dataset files into a single NetCDF file.
Based on the dataset configuration: 128x384 resolution, 101 timesteps.
Fields: density, pressure (scalars), velocity_x, velocity_y (vector components)
Constant fields: tcool (ignored in assembler)
"""

import argparse
import os
import glob
from netCDF4 import Dataset
import numpy as np


def assemble_turbulent_radiative_data(input_dir, output_file):
    """Assemble all turbulent radiative layer 2D files into a single NetCDF file."""
    
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
        print(f"Groups in first file: {list(first_nc.groups.keys())}")
        
        # Get dimensions from density field in t0_fields
        density_var = first_nc.groups['t0_fields'].variables['density']
        n_trajectories = density_var.shape[0]
        n_timesteps = density_var.shape[1] 
        height = density_var.shape[2]  # y dimension (128)
        width = density_var.shape[3]   # x dimension (384)
        
        samples.append(n_trajectories)
        
        print(f"File structure: {n_trajectories} trajectories, {n_timesteps} timesteps, {height}x{width} resolution")
    
    # Count samples from all files
    for nc_file in nc_files[1:]:
        with Dataset(nc_file, "r") as nc:
            density_var = nc.groups['t0_fields'].variables['density']
            samples.append(density_var.shape[0])
    
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
        
        # Create variables for each field (ignoring constant fields like tcool)
        print("Creating variables...")
        
        # Scalar fields: density, pressure (sample, time, y, x)
        density_var = out_nc.createVariable(
            "density", "f4", ("sample", "time", "y", "x"),
            chunksizes=(1, 1, height, width)
        )
        
        pressure_var = out_nc.createVariable(
            "pressure", "f4", ("sample", "time", "y", "x"),
            chunksizes=(1, 1, height, width)
        )
        
        # Vector field: velocity (sample, time, y, x, vector_dim)
        velocity_var = out_nc.createVariable(
            "velocity", "f4", ("sample", "time", "y", "x", "vector_dim"),
            chunksizes=(1, 1, height, width, 2)
        )
        
        # Copy data from all files
        for i, nc_file in enumerate(nc_files):
            print(f"Processing {nc_file} ({i+1}/{len(nc_files)})")
            
            with Dataset(nc_file, "r") as nc:
                # Read data from groups
                density_data = nc.groups['t0_fields'].variables['density'][:]
                pressure_data = nc.groups['t0_fields'].variables['pressure'][:]
                velocity_data = nc.groups['t1_fields'].variables['velocity'][:]
                
                # Write to output file
                start_idx = samples[i]
                end_idx = samples[i + 1]
                
                density_var[start_idx:end_idx] = density_data
                pressure_var[start_idx:end_idx] = pressure_data
                velocity_var[start_idx:end_idx] = velocity_data
        
        # Add metadata
        out_nc.setncattr("description", "Assembled Well Turbulent Radiative Layer 2D dataset")
        out_nc.setncattr("total_samples", total_samples)
        out_nc.setncattr("timesteps", n_timesteps)
        out_nc.setncattr("spatial_resolution", f"{height}x{width}")
        out_nc.setncattr("fields", "density,pressure,velocity_x,velocity_y")
        out_nc.setncattr("dataset_name", "turbulent_radiative_layer_2D")
        out_nc.setncattr("boundary_conditions", "OPEN,PERIODIC")
    
    print(f"Successfully assembled data to {output_file}")
    print(f"Final structure:")
    print(f"  - {total_samples} samples")
    print(f"  - {n_timesteps} timesteps")  
    print(f"  - {height}x{width} spatial resolution")
    print(f"  - 4 fields: density, pressure, velocity_x, velocity_y")


def main():
    parser = argparse.ArgumentParser(description="Assemble Well Turbulent Radiative Layer 2D dataset")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing turbulent radiative layer 2D .hdf5 files")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output assembled .nc file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    assemble_turbulent_radiative_data(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
