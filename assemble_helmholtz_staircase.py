#!/usr/bin/env python3
"""
Assemble Well Helmholtz Staircase dataset files into a single NetCDF file.
Based on the dataset configuration: 1024x256 resolution, 50 timesteps.
Fields: pressure_re, pressure_im (complex pressure field components)
Constant fields: mask, omega (ignored in assembler - mask is constant field)
"""

import argparse
import os
import glob
from netCDF4 import Dataset
import numpy as np


def assemble_helmholtz_staircase_data(input_dir, output_file):
    """Assemble all Helmholtz staircase files into a single NetCDF file."""
    
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
        out_nc.createDimension("complex_dim", 2)  # For real and imaginary parts
        
        # Create variables for each field (ignoring constant fields omega, mask)
        print("Creating variables...")
        
        # Complex pressure field: pressure_re, pressure_im -> combined (sample, time, y, x, complex_dim)
        pressure_var = out_nc.createVariable(
            "pressure", "f4", ("sample", "time", "y", "x", "complex_dim"),
            chunksizes=(1, 1, height, width, 2)
        )
        
        # Alternative: Keep as separate real and imaginary components
        pressure_re_var = out_nc.createVariable(
            "pressure_re", "f4", ("sample", "time", "y", "x"),
            chunksizes=(1, 1, height, width)
        )
        
        pressure_im_var = out_nc.createVariable(
            "pressure_im", "f4", ("sample", "time", "y", "x"),
            chunksizes=(1, 1, height, width)
        )
        
        # Copy data from all files
        for i, nc_file in enumerate(nc_files):
            print(f"Processing {nc_file} ({i+1}/{len(nc_files)})")
            
            with Dataset(nc_file, "r") as nc:
                # Read data for each field
                pressure_re_data = nc.variables['pressure_re'][:]
                pressure_im_data = nc.variables['pressure_im'][:]
                
                # Combine real and imaginary parts
                pressure_complex_data = np.stack([pressure_re_data, pressure_im_data], axis=-1)
                
                # Write to output file
                start_idx = samples[i]
                end_idx = samples[i + 1]
                
                pressure_var[start_idx:end_idx] = pressure_complex_data
                pressure_re_var[start_idx:end_idx] = pressure_re_data
                pressure_im_var[start_idx:end_idx] = pressure_im_data
        
        # Add metadata
        out_nc.setncattr("description", "Assembled Well Helmholtz Staircase dataset")
        out_nc.setncattr("total_samples", total_samples)
        out_nc.setncattr("timesteps", n_timesteps)
        out_nc.setncattr("spatial_resolution", f"{height}x{width}")
        out_nc.setncattr("fields", "pressure_re,pressure_im,pressure")
        out_nc.setncattr("dataset_name", "helmholtz_staircase")
        out_nc.setncattr("boundary_conditions", "OPEN,WALL")
        out_nc.setncattr("equation_type", "helmholtz")
        out_nc.setncattr("field_type", "complex_pressure")
    
    print(f"Successfully assembled data to {output_file}")
    print(f"Final structure:")
    print(f"  - {total_samples} samples")
    print(f"  - {n_timesteps} timesteps")  
    print(f"  - {height}x{width} spatial resolution")
    print(f"  - Complex pressure field: pressure_re, pressure_im (+ combined pressure)")


def main():
    parser = argparse.ArgumentParser(description="Assemble Well Helmholtz Staircase dataset")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing Helmholtz staircase .nc files")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output assembled .nc file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    assemble_helmholtz_staircase_data(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
