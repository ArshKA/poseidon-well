import os
import torch
import numpy as np
import netCDF4
import yaml
from ..base import BaseTimeDataset


class WellActiveMatter(BaseTimeDataset):
    """Simplified Well Active Matter dataset using assembled NetCDF file."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Time constraint - we have 81 actual timesteps per trajectory (0 to 80)
        assert self.max_num_time_steps * self.time_step_size <= 81
        
        # Dataset parameters
        self.N_max = 10000  # Large enough to accommodate all data
        self.N_val = 100
        self.N_test = 200
        
        # Data specifications
        self.resolution = 256
        self.input_dim = 11  # 1 (concentration) + 2 (velocity) + 4 (D) + 4 (E)
        self.output_dim = 11  # Same as input
        self.label_description = "[concentration],[velocity_x,velocity_y],[D_00,D_01,D_10,D_11],[E_00,E_01,E_10,E_11]"
        
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
                f"Please run: python assemble_well_active_matter.py "
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
                "time": 20.0
            }
        
        with open(stats_file, 'r') as f:
            stats = yaml.safe_load(f)
        
        # Convert to torch tensors with proper shapes for broadcasting
        constants = {
            "time": 20.0,  # Time goes from 0 to 20
        }
        
        # Process each field's normalization
        field_shapes = {
            'concentration': (1,),  # scalar
            'velocity': (2,),       # 2D vector
            'D': (4,),             # flattened 2x2 tensor
            'E': (4,)              # flattened 2x2 tensor
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
                    # Vector/tensor field
                    if field in ['D', 'E']:
                        # Flatten 2x2 tensor stats
                        flat_mean = np.array(field_mean).flatten()
                        flat_std = np.array(field_std).flatten()
                        mean_values.extend(flat_mean.tolist())
                        std_values.extend(flat_std.tolist())
                    else:
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
        # Each trajectory provides a number of possible starting time windows.
        # If a trajectory has 81 steps (0-80) and we need a window of size `T`,
        # the last possible start time is 81 - T. So there are 81 - T + 1 windows.
        # We assume `T = max_num_time_steps * time_step_size`.
        window_size = self.max_num_time_steps * self.time_step_size
        timesteps_per_trajectory = 81 - window_size
        
        # We need to predict one step ahead, so the input can be at most at t=80-time_step_size
        timesteps_per_trajectory = 81 - self.time_step_size
        
        # The number of possible starting points for a single time step prediction
        timesteps_per_sample = 81 - self.time_step_size
        return self.num_trajectories * timesteps_per_sample

    def __getitem__(self, idx):
        """Load a single sample corresponding to a linear index."""
        
        # ### --- CORRECTED MAPPING LOGIC --- ###
        # The total number of valid starting time points in a single trajectory.
        # The latest input can be at t=80-time_step_size to predict t=80.
        timesteps_per_sample = 81 - self.time_step_size
        if timesteps_per_sample <= 0:
            raise ValueError(
                f"time_step_size ({self.time_step_size}) is too large for the "
                f"available 81 timesteps."
            )

        # Correctly map the linear index `idx` to a trajectory and a time window.
        # `sample_idx` is which trajectory to use.
        # `time_offset` is the starting time point within that trajectory.
        sample_idx = idx // timesteps_per_sample
        time_offset = idx % timesteps_per_sample

        # Define the actual start and end time indices based on the offset.
        actual_t1 = time_offset                # Input time
        actual_t2 = time_offset + self.time_step_size  # Target time
        
        # Defensive check to prevent out-of-bounds access before I/O
        if actual_t2 > 80:
            raise IndexError(
                f"Calculated target time index {actual_t2} is out of bounds for idx {idx}. "
                f"Max timestep is 80."
            )
        # ### --- END OF CORRECTION --- ###

        try:
            with netCDF4.Dataset(self.data_file, 'r') as dataset:
                # Load input fields at time actual_t1
                conc_input = dataset.variables['concentration'][sample_idx, actual_t1, :, :]  # (y, x)
                vel_input = dataset.variables['velocity'][sample_idx, actual_t1, :, :, :]      # (y, x, 2)
                d_input = dataset.variables['D'][sample_idx, actual_t1, :, :, :]               # (y, x, 4)
                e_input = dataset.variables['E'][sample_idx, actual_t1, :, :, :]               # (y, x, 4)
                
                # Load target fields at time actual_t2
                conc_target = dataset.variables['concentration'][sample_idx, actual_t2, :, :]  # (y, x)
                vel_target = dataset.variables['velocity'][sample_idx, actual_t2, :, :, :]      # (y, x, 2)
                d_target = dataset.variables['D'][sample_idx, actual_t2, :, :, :]               # (y, x, 4)
                e_target = dataset.variables['E'][sample_idx, actual_t2, :, :, :]               # (y, x, 4)
                
                # Reshape and concatenate inputs: (y, x) -> (1, y, x), (y, x, n) -> (n, y, x)
                inputs = torch.cat([
                    torch.from_numpy(conc_input.astype(np.float32)).unsqueeze(0),  # (1, y, x)
                    torch.from_numpy(vel_input.astype(np.float32)).permute(2, 0, 1),  # (2, y, x)
                    torch.from_numpy(d_input.astype(np.float32)).permute(2, 0, 1),    # (4, y, x)
                    torch.from_numpy(e_input.astype(np.float32)).permute(2, 0, 1)     # (4, y, x)
                ], dim=0)  # (11, y, x)
                
                # Reshape and concatenate targets
                labels = torch.cat([
                    torch.from_numpy(conc_target.astype(np.float32)).unsqueeze(0),  # (1, y, x)
                    torch.from_numpy(vel_target.astype(np.float32)).permute(2, 0, 1),  # (2, y, x)
                    torch.from_numpy(d_target.astype(np.float32)).permute(2, 0, 1),    # (4, y, x)
                    torch.from_numpy(e_target.astype(np.float32)).permute(2, 0, 1)     # (4, y, x)
                ], dim=0)  # (11, y, x)
                
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