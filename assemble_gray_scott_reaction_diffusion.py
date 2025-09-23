#!/usr/bin/env python3
"""
Assemble Well Gray Scott Reaction Diffusion dataset files into a single NetCDF file.
Based on the dataset configuration: 128x128 resolution, 1001 timesteps.
Fields: A, B (scalar concentration fields)
Constant fields: F, k (ignored in assembler)
"""

import argparse
import os
import glob
from netCDF4 import Dataset
import numpy as np


def assemble_gray_scott_data(input_dir, output_file):
    """Assemble all Gray Scott reaction diffusion files into a single NetCDF file."""
    
    # Get all .nc files
    pattern = os.path.join(input_dir, "**", "*.nc")
    nc_files = glob.glob(pattern, recursive=True)
    nc_files.sort()
    
    print(f"Found {len(nc_files)} files to assemble")
    
    if not nc_files:
        raise ValueError(f"No .nc files found in {input_dir}")
    
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
        
        # Create variables for each field (ignoring constant fields F, k)
        print("Creating variables...")
        
        # Scalar fields: A, B (sample, time, y, x)
        a_var = out_nc.createVariable(
            "A", "f4", ("sample", "time", "y", "x"),
            chunksizes=(1, 1, height, width)
        )
        
        b_var = out_nc.createVariable(
            "B", "f4", ("sample", "time", "y", "x"),
            chunksizes=(1, 1, height, width)
        )
        
        # Copy data from all files
        for i, nc_file in enumerate(nc_files):
            print(f"Processing {nc_file} ({i+1}/{len(nc_files)})")
            
            with Dataset(nc_file, "r") as nc:
                # Read data for each field
                a_data = nc.variables['A'][:]
                b_data = nc.variables['B'][:]
                
                # Write to output file
                start_idx = samples[i]
                end_idx = samples[i + 1]
                
                a_var[start_idx:end_idx] = a_data
                b_var[start_idx:end_idx] = b_data
        
        # Add metadata
        out_nc.setncattr("description", "Assembled Well Gray Scott Reaction Diffusion dataset")
        out_nc.setncattr("total_samples", total_samples)
        out_nc.setncattr("timesteps", n_timesteps)
        out_nc.setncattr("spatial_resolution", f"{height}x{width}")
        out_nc.setncattr("fields", "A,B")
        out_nc.setncattr("dataset_name", "gray_scott_reaction_diffusion")
        out_nc.setncattr("boundary_conditions", "PERIODIC")
        out_nc.setncattr("equation_type", "reaction_diffusion")
    
    print(f"Successfully assembled data to {output_file}")
    print(f"Final structure:")
    print(f"  - {total_samples} samples")
    print(f"  - {n_timesteps} timesteps")  
    print(f"  - {height}x{width} spatial resolution")
    print(f"  - 2 fields: A, B (concentration fields)")


def main():
    parser = argparse.ArgumentParser(description="Assemble Well Gray Scott Reaction Diffusion dataset")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing Gray Scott reaction diffusion .nc files")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output assembled .nc file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    assemble_gray_scott_data(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
