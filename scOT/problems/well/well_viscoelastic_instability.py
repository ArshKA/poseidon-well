import os
import torch
import numpy as np
import netCDF4
import yaml
from ..base import BaseTimeDataset


class WellViscoelasticInstability(BaseTimeDataset):
    """Well Viscoelastic Instability dataset using assembled NetCDF file."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Time constraint - we have varying timesteps (20-60) per trajectory, use 60 as max
        assert self.max_num_time_steps * self.time_step_size <= 60
        
        # Dataset parameters
        self.N_max = 10000  # Large enough to accommodate all data
        self.N_val = 100
        self.N_test = 200
        
        # Data specifications
        self.resolution = 512  # 512x512 resolution
        self.input_dim = 8  # pressure, c_zz, velocity(2), C(4)
        self.output_dim = 8  # Same as input
        self.label_description = "[pressure],[c_zz],[velocity_x,velocity_y],[C_0,C_1,C_2,C_3]"
        
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
                f"Please run: python assemble_viscoelastic_instability.py "
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
                "time": 59.0
            }
        
        with open(stats_file, 'r') as f:
            stats = yaml.safe_load(f)
        
        constants = {
            "time": 59.0,  # Time goes from 0 to 59 (max timesteps)
        }
        
        means = stats.get('mean', {})
        stds = stats.get('std', {})
        
        mean_values, std_values = [], []

        def _get_stat(stats_dict, key, default, length):
            val = stats_dict.get(key, default)
            if isinstance(val, (int, float)):
                return [float(val)] * length
            flat_val = np.array(val, dtype=np.float32).flatten().tolist()
            while len(flat_val) < length:
                flat_val.append(float(default))
            return flat_val[:length]

        mean_values.extend(_get_stat(means, 'pressure', 0.0, 1))
        std_values.extend(_get_stat(stds, 'pressure', 1.0, 1))
        
        mean_values.extend(_get_stat(means, 'c_zz', 0.0, 1))
        std_values.extend(_get_stat(stds, 'c_zz', 1.0, 1))

        mean_values.extend(_get_stat(means, 'velocity', 0.0, 2))
        std_values.extend(_get_stat(stds, 'velocity', 1.0, 2))

        mean_values.extend(_get_stat(means, 'C', 0.0, 4))
        std_values.extend(_get_stat(stds, 'C', 1.0, 4))
        
        constants["mean"] = torch.tensor(mean_values, dtype=torch.float32).reshape(-1, 1, 1)
        constants["std"] = torch.tensor(std_values, dtype=torch.float32).reshape(-1, 1, 1)
        
        return constants
    
    def _calculate_dataset_size(self):
        """Calculate the total number of trajectories from the assembled file."""
        try:
            with netCDF4.Dataset(self.data_file, 'r') as dataset:
                return dataset.dimensions['sample'].size
        except Exception as e:
            print(f"Warning: Could not read dataset size from {self.data_file}: {e}")
            return 100  # Fallback
    
    def __len__(self):
        """Return the total number of time-dependent samples based on max trajectory length."""
        # If there are max 60 timesteps (0-59), the last input can be at t=59-time_step_size.
        timesteps_per_sample = 60 - self.time_step_size
        return self.num_trajectories * timesteps_per_sample
    
    def __getitem__(self, idx):
        """Load a single sample corresponding to a linear index."""

        # ### --- CORRECTED MAPPING LOGIC --- ###
        timesteps_per_sample = 60 - self.time_step_size
        if timesteps_per_sample <= 0:
            raise ValueError(
                f"time_step_size ({self.time_step_size}) is too large for the "
                f"available 60 timesteps."
            )

        sample_idx = idx // timesteps_per_sample
        time_offset = idx % timesteps_per_sample
        actual_t1 = time_offset
        actual_t2 = time_offset + self.time_step_size

        if actual_t2 > 59:
            raise IndexError(
                f"Calculated target time index {actual_t2} is out of bounds for idx {idx}. "
                f"Max timestep is 59."
            )
        # ### --- END OF CORRECTION --- ###
        
        try:
            with netCDF4.Dataset(self.data_file, 'r') as dataset:
                # Load input fields at time actual_t1
                pressure_input = dataset.variables['pressure'][sample_idx, actual_t1, :, :]
                c_zz_input = dataset.variables['c_zz'][sample_idx, actual_t1, :, :]
                velocity_input = dataset.variables['velocity'][sample_idx, actual_t1, :, :, :]
                C_input = dataset.variables['C'][sample_idx, actual_t1, :, :, :]
                
                # Load target fields at time actual_t2
                pressure_target = dataset.variables['pressure'][sample_idx, actual_t2, :, :]
                c_zz_target = dataset.variables['c_zz'][sample_idx, actual_t2, :, :]
                velocity_target = dataset.variables['velocity'][sample_idx, actual_t2, :, :, :]
                C_target = dataset.variables['C'][sample_idx, actual_t2, :, :, :]
                
                # Reshape and concatenate inputs
                inputs = torch.cat([
                    torch.from_numpy(pressure_input.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(c_zz_input.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(velocity_input.astype(np.float32)).permute(2, 0, 1), # (y, x, 2) -> (2, y, x)
                    torch.from_numpy(C_input.astype(np.float32)).permute(2, 0, 1),      # (y, x, 4) -> (4, y, x)
                ], dim=0)
                
                # Reshape and concatenate targets
                labels = torch.cat([
                    torch.from_numpy(pressure_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(c_zz_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(velocity_target.astype(np.float32)).permute(2, 0, 1),
                    torch.from_numpy(C_target.astype(np.float32)).permute(2, 0, 1),
                ], dim=0)
                
        except Exception as e:
            print(f"Error loading sample idx={idx} (sample_idx={sample_idx}, t1={actual_t1}, t2={actual_t2}): {e}")
            # Return dummy data to prevent training crashes
            inputs = torch.zeros(self.input_dim, self.resolution, self.resolution)
            labels = torch.zeros(self.input_dim, self.resolution, self.resolution)
        
        # Apply normalization
        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]
        
        # Normalize time
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