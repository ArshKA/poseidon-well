import os
import torch
import numpy as np
import netCDF4
import yaml
from ..base import BaseTimeDataset


class WellGrayScottReactionDiffusion(BaseTimeDataset):
    """Well Gray Scott Reaction Diffusion dataset using assembled NetCDF file."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Time constraint - we have 1001 actual timesteps per trajectory (0 to 1000)
        assert self.max_num_time_steps * self.time_step_size <= 1001
        
        # Dataset parameters
        self.N_max = 10000  # Large enough to accommodate all data
        self.N_val = 100
        self.N_test = 200
        
        # Data specifications
        self.resolution = 128  # 128x128 resolution
        self.input_dim = 2  # A + B (concentration fields)
        self.output_dim = 2  # Same as input
        self.label_description = "[A],[B]"
        
        # Find assembled data file
        self.data_file = self._find_assembled_file()
        
        # Load normalization constants
        self.constants = self._load_normalization_constants()
        
        # Calculate actual dataset size
        self.num_trajectories = self._calculate_dataset_size()
        
        # Validate dataset
        self.post_init()
    
    def _find_assembled_file(self):
        """Find the assembled NetCDF file."""
        assembled_file = os.path.join(self.data_path, f"assembled_{self.which}.nc")
        
        if not os.path.exists(assembled_file):
            raise FileNotFoundError(
                f"Assembled file not found: {assembled_file}\n"
                f"Please run: python assemble_gray_scott_reaction_diffusion.py "
                f"--input_dir {self.data_path}/data/{self.which} "
                f"--output_file {assembled_file}"
            )
        
        return assembled_file
    
    def _load_normalization_constants(self):
        """Load normalization constants from stats.yaml."""
        stats_file = os.path.join(self.data_path, "stats.yaml")
        
        if not os.path.exists(stats_file):
            print(f"Warning: stats.yaml not found at {stats_file}, using default normalization")
            return {
                "mean": torch.zeros(self.input_dim, 1, 1),
                "std": torch.ones(self.input_dim, 1, 1),
                "time": 1000.0
            }
        
        with open(stats_file, 'r') as f:
            stats = yaml.safe_load(f)
        
        # Convert to torch tensors with proper shapes for broadcasting
        constants = {
            "time": 1000.0,  # Time goes from 0 to 1000
        }
        
        # Process each field's normalization
        field_shapes = {
            'A': (1,),  # scalar concentration
            'B': (1,),  # scalar concentration
        }
        
        mean_values = []
        std_values = []
        
        # Extract mean and std sections from YAML
        means = stats.get('mean', {})
        stds = stats.get('std', {})
        
        for field, shape in field_shapes.items():
            if field in means and field in stds:
                field_mean = means[field]
                field_std = stds[field]
                
                if isinstance(field_mean, (int, float)):
                    # Scalar field
                    mean_values.extend([field_mean] * shape[0])
                    std_values.extend([field_std] * shape[0])
                else:
                    # Vector field
                    mean_values.extend(field_mean)
                    std_values.extend(field_std)
            else:
                # Default normalization for missing fields
                print(f"Warning: No normalization constants found for field '{field}', using defaults")
                mean_values.extend([0.0] * shape[0])
                std_values.extend([1.0] * shape[0])
        
        # Convert to tensors with shape (channels, 1, 1) for broadcasting
        constants["mean"] = torch.tensor(mean_values, dtype=torch.float32).reshape(-1, 1, 1)
        constants["std"] = torch.tensor(std_values, dtype=torch.float32).reshape(-1, 1, 1)
        
        return constants
    
    def _calculate_dataset_size(self):
        """Calculate the total number of trajectories from the assembled file."""
        try:
            with netCDF4.Dataset(self.data_file, 'r') as dataset:
                total_samples = dataset.dimensions['sample'].size
                return total_samples
        except Exception as e:
            print(f"Warning: Could not read dataset size from {self.data_file}: {e}")
            return 100  # Fallback
    
    def __len__(self):
        """Return the total number of time-dependent samples."""
        # The number of possible starting points for a single time step prediction.
        # If there are 1001 timesteps (0-1000), the last input can be at t=1000-time_step_size.
        timesteps_per_sample = 1001 - self.time_step_size
        return self.num_trajectories * timesteps_per_sample
    
    def __getitem__(self, idx):
        """Load a single sample corresponding to a linear index."""
        
        # ### --- CORRECTED MAPPING LOGIC --- ###
        # The total number of valid starting time points in a single trajectory.
        timesteps_per_sample = 1001 - self.time_step_size
        if timesteps_per_sample <= 0:
            raise ValueError(
                f"time_step_size ({self.time_step_size}) is too large for the "
                f"available 1001 timesteps."
            )

        # Correctly map the linear index `idx` to a trajectory and a time window.
        sample_idx = idx // timesteps_per_sample
        time_offset = idx % timesteps_per_sample

        # Define the actual start and end time indices based on the offset.
        actual_t1 = time_offset                # Input time
        actual_t2 = time_offset + self.time_step_size  # Target time
        
        # Defensive check to prevent out-of-bounds access before I/O
        if actual_t2 > 1000:
            raise IndexError(
                f"Calculated target time index {actual_t2} is out of bounds for idx {idx}. "
                f"Max timestep is 1000."
            )
        # ### --- END OF CORRECTION --- ###
        
        try:
            with netCDF4.Dataset(self.data_file, 'r') as dataset:
                # Load input fields at time actual_t1
                a_input = dataset.variables['A'][sample_idx, actual_t1, :, :]  # (y, x)
                b_input = dataset.variables['B'][sample_idx, actual_t1, :, :]  # (y, x)
                
                # Load target fields at time actual_t2
                a_target = dataset.variables['A'][sample_idx, actual_t2, :, :]  # (y, x)
                b_target = dataset.variables['B'][sample_idx, actual_t2, :, :]  # (y, x)
                
                # Reshape and concatenate inputs: (y, x) -> (1, y, x)
                inputs = torch.stack([
                    torch.from_numpy(a_input.astype(np.float32)),
                    torch.from_numpy(b_input.astype(np.float32)),
                ], dim=0)  # (2, y, x)
                
                # Reshape and concatenate targets
                labels = torch.stack([
                    torch.from_numpy(a_target.astype(np.float32)),
                    torch.from_numpy(b_target.astype(np.float32)),
                ], dim=0)  # (2, y, x)
                
        except Exception as e:
            print(f"Error loading sample idx={idx} (sample_idx={sample_idx}, t1={actual_t1}, t2={actual_t2}): {e}")
            # Return dummy data to prevent training crashes
            inputs = torch.zeros(self.input_dim, self.resolution, self.resolution)
            labels = torch.zeros(self.input_dim, self.resolution, self.resolution)
        
        # Apply normalization
        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]
        
        # Normalize time (using the input time)
        time_normalized = actual_t1 / self.constants["time"]
        
        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time_normalized
        }
    
    def denormalize(self, data):
        """
        Denormalize data back to original scale.
        
        Args:
            data: Tensor or numpy array with shape (batch_size, channels, height, width) or (channels, height, width)
            
        Returns:
            Denormalized data in the same format as input
        """
        if isinstance(data, torch.Tensor):
            mean = self.constants["mean"].to(data.device)
            std = self.constants["std"].to(data.device)
            return data * std + mean
        elif isinstance(data, np.ndarray):
            mean = self.constants["mean"].cpu().numpy()
            std = self.constants["std"].cpu().numpy()
            return data * std + mean
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def get_normalization_constants(self):
        """
        Get normalization constants for external use.
        
        Returns:
            Dictionary containing mean and std tensors
        """
        return {
            "mean": self.constants["mean"],
            "std": self.constants["std"],
            "time": self.constants["time"]
        }