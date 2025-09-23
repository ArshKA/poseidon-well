#!/usr/bin/env python3
"""
Master script to assemble all Well datasets using their respective assemblers.
This script provides a unified interface to run all dataset assemblers.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Dataset configurations mapping dataset name to assembler script
DATASET_ASSEMBLERS = {
    "shear_flow": "assemble_shear_flow.py",
    "rayleigh_benard": "assemble_rayleigh_benard.py", 
    "acoustic_scattering_maze": "assemble_acoustic_scattering_maze.py",
    "active_matter": "assemble_active_matter_well.py",
    "turbulent_radiative_layer_2D": "assemble_turbulent_radiative_layer_2D.py",
    "viscoelastic_instability": "assemble_viscoelastic_instability.py",
    "gray_scott_reaction_diffusion": "assemble_gray_scott_reaction_diffusion.py",
    "helmholtz_staircase": "assemble_helmholtz_staircase.py"
}

# Dataset properties for reference
DATASET_INFO = {
    "shear_flow": {
        "resolution": "256x512",
        "timesteps": 200,
        "fields": ["tracer", "pressure", "velocity_x", "velocity_y"],
        "description": "2D periodic incompressible shear flow"
    },
    "rayleigh_benard": {
        "resolution": "512x128", 
        "timesteps": 200,
        "fields": ["buoyancy", "pressure", "velocity_x", "velocity_y"],
        "description": "Rayleigh-Bénard convection"
    },
    "acoustic_scattering_maze": {
        "resolution": "256x256",
        "timesteps": 202,
        "fields": ["pressure", "velocity_x", "velocity_y"],
        "description": "Acoustic scattering in maze geometry"
    },
    "active_matter": {
        "resolution": "256x256",
        "timesteps": 81,
        "fields": ["concentration", "velocity", "D_tensor", "E_tensor"],
        "description": "Active matter dynamics with concentration and tensor fields"
    },
    "turbulent_radiative_layer_2D": {
        "resolution": "128x384",
        "timesteps": 101,
        "fields": ["density", "pressure", "velocity_x", "velocity_y"],
        "description": "2D turbulent radiative layer"
    },
    "viscoelastic_instability": {
        "resolution": "512x512",
        "timesteps": "20-60",
        "fields": ["pressure", "c_zz", "velocity", "C_tensor"],
        "description": "Viscoelastic instability with tensor stress fields"
    },
    "gray_scott_reaction_diffusion": {
        "resolution": "128x128",
        "timesteps": 1001,
        "fields": ["A", "B"],
        "description": "Gray-Scott reaction-diffusion system"
    },
    "helmholtz_staircase": {
        "resolution": "1024x256",
        "timesteps": 50,
        "fields": ["pressure_re", "pressure_im"],
        "description": "Helmholtz equation with complex pressure field"
    }
}


def list_datasets():
    """List all available datasets with their properties."""
    print("Available Well Datasets:")
    print("=" * 80)
    
    for dataset, info in DATASET_INFO.items():
        print(f"\nDataset: {dataset}")
        print(f"  Resolution: {info['resolution']}")
        print(f"  Timesteps: {info['timesteps']}")
        print(f"  Fields: {', '.join(info['fields'])}")
        print(f"  Description: {info['description']}")
        print(f"  Assembler: {DATASET_ASSEMBLERS[dataset]}")


def run_assembler(dataset_name, input_dir, output_file, script_dir=None):
    """Run the assembler for a specific dataset."""
    
    if dataset_name not in DATASET_ASSEMBLERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_ASSEMBLERS.keys())}")
    
    assembler_script = DATASET_ASSEMBLERS[dataset_name]
    
    # Determine script path
    if script_dir:
        script_path = os.path.join(script_dir, assembler_script)
    else:
        script_path = os.path.join(os.path.dirname(__file__), assembler_script)
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Assembler script not found: {script_path}")
    
    # Build command
    cmd = [sys.executable, script_path, "--input_dir", input_dir, "--output_file", output_file]
    
    print(f"Running assembler for {dataset_name}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the assembler
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Assembly completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running assembler: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Assemble Well datasets using their respective assemblers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python assemble_all_well_datasets.py --list
  
  # Assemble a specific dataset
  python assemble_all_well_datasets.py --dataset shear_flow --input_dir /path/to/data --output_file /path/to/output.nc
  
  # Assemble a dataset
  python assemble_all_well_datasets.py --dataset active_matter --input_dir /data/active_matter --output_file active_matter.nc
        """
    )
    
    parser.add_argument("--list", action="store_true",
                       help="List all available datasets and their properties")
    parser.add_argument("--dataset", type=str,
                       help="Dataset name to assemble")
    parser.add_argument("--input_dir", type=str,
                       help="Directory containing dataset files")
    parser.add_argument("--output_file", type=str,
                       help="Output assembled .nc file")
    parser.add_argument("--script_dir", type=str,
                       help="Directory containing assembler scripts (default: same as this script)")
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if not args.dataset:
        print("Error: --dataset is required (use --list to see available datasets)")
        parser.print_help()
        return
    
    if not args.input_dir:
        print("Error: --input_dir is required")
        parser.print_help()
        return
        
    if not args.output_file:
        print("Error: --output_file is required")
        parser.print_help()
        return
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Run the assembler
    success = run_assembler(
        args.dataset, 
        args.input_dir, 
        args.output_file, 
        args.script_dir
    )
    
    if success:
        print(f"\n✅ Successfully assembled {args.dataset} dataset!")
        print(f"Output file: {args.output_file}")
    else:
        print(f"\n❌ Failed to assemble {args.dataset} dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()
