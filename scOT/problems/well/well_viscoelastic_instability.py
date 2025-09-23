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
        self.input_dim = 8  # pressure + c_zz + velocity_x + velocity_y + C_xx + C_xy + C_yx + C_yy
        self.output_dim = 8  # Same as input
        self.label_description = "[pressure],[c_zz],[velocity_x],[velocity_y],[C_xx],[C_xy],[C_yx],[C_yy]"
        
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
                "time": 60.0
            }
        
        with open(stats_file, 'r') as f:
            stats = yaml.safe_load(f)
        
        # Convert to torch tensors with proper shapes for broadcasting
        constants = {
            "time": 60.0,  # Time goes from 0 to 60 (max timesteps)
        }
        
        # Process each field's normalization
        field_shapes = {
            'pressure': (1,),    # scalar
            'c_zz': (1,),        # scalar
            'velocity_x': (1,),  # scalar
            'velocity_y': (1,),  # scalar
            'C_xx': (1,),        # tensor component
            'C_xy': (1,),        # tensor component
            'C_yx': (1,),        # tensor component
            'C_yy': (1,),        # tensor component
        }
        
        mean_values = []
        std_values = []
        
        for field, shape in field_shapes.items():
            if field in stats:
                field_mean = stats[field]['mean']
                field_std = stats[field]['std']
                
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
        # Each sample can provide (n_timesteps - max_num_time_steps * time_step_size + 1) time samples
        timesteps_per_sample = 60 - self.max_num_time_steps * self.time_step_size + 1
        return self.num_trajectories * timesteps_per_sample
    
    def __getitem__(self, idx):
        """Load a single sample."""
        # Map linear index to time-dependent sample
        i, t, t1, t2 = self._idx_map(idx)
        
        # Map to actual sample and time indices
        timesteps_per_sample = 60 - self.max_num_time_steps * self.time_step_size + 1
        sample_idx = i % self.num_trajectories  # Which trajectory
        time_offset = i // self.num_trajectories  # Which time window within trajectory
        
        # Adjust time indices
        actual_t1 = t1 + time_offset
        actual_t2 = t2 + time_offset
        
        try:
            with netCDF4.Dataset(self.data_file, 'r') as dataset:
                # Load input fields at time actual_t1
                pressure_input = dataset.variables['pressure'][sample_idx, actual_t1, :, :]   # (y, x)
                c_zz_input = dataset.variables['c_zz'][sample_idx, actual_t1, :, :]           # (y, x)
                vel_x_input = dataset.variables['velocity_x'][sample_idx, actual_t1, :, :]    # (y, x)
                vel_y_input = dataset.variables['velocity_y'][sample_idx, actual_t1, :, :]    # (y, x)
                c_xx_input = dataset.variables['C_xx'][sample_idx, actual_t1, :, :]           # (y, x)
                c_xy_input = dataset.variables['C_xy'][sample_idx, actual_t1, :, :]           # (y, x)
                c_yx_input = dataset.variables['C_yx'][sample_idx, actual_t1, :, :]           # (y, x)
                c_yy_input = dataset.variables['C_yy'][sample_idx, actual_t1, :, :]           # (y, x)
                
                # Load target fields at time actual_t2
                pressure_target = dataset.variables['pressure'][sample_idx, actual_t2, :, :]   # (y, x)
                c_zz_target = dataset.variables['c_zz'][sample_idx, actual_t2, :, :]           # (y, x)
                vel_x_target = dataset.variables['velocity_x'][sample_idx, actual_t2, :, :]    # (y, x)
                vel_y_target = dataset.variables['velocity_y'][sample_idx, actual_t2, :, :]    # (y, x)
                c_xx_target = dataset.variables['C_xx'][sample_idx, actual_t2, :, :]           # (y, x)
                c_xy_target = dataset.variables['C_xy'][sample_idx, actual_t2, :, :]           # (y, x)
                c_yx_target = dataset.variables['C_yx'][sample_idx, actual_t2, :, :]           # (y, x)
                c_yy_target = dataset.variables['C_yy'][sample_idx, actual_t2, :, :]           # (y, x)
                
                # Reshape and concatenate inputs: (y, x) -> (1, y, x)
                inputs = torch.cat([
                    torch.from_numpy(pressure_input.astype(np.float32)).unsqueeze(0),  # (1, y, x)
                    torch.from_numpy(c_zz_input.astype(np.float32)).unsqueeze(0),     # (1, y, x)
                    torch.from_numpy(vel_x_input.astype(np.float32)).unsqueeze(0),    # (1, y, x)
                    torch.from_numpy(vel_y_input.astype(np.float32)).unsqueeze(0),    # (1, y, x)
                    torch.from_numpy(c_xx_input.astype(np.float32)).unsqueeze(0),     # (1, y, x)
                    torch.from_numpy(c_xy_input.astype(np.float32)).unsqueeze(0),     # (1, y, x)
                    torch.from_numpy(c_yx_input.astype(np.float32)).unsqueeze(0),     # (1, y, x)
                    torch.from_numpy(c_yy_input.astype(np.float32)).unsqueeze(0),     # (1, y, x)
                ], dim=0)  # (8, y, x)
                
                # Reshape and concatenate targets
                labels = torch.cat([
                    torch.from_numpy(pressure_target.astype(np.float32)).unsqueeze(0),  # (1, y, x)
                    torch.from_numpy(c_zz_target.astype(np.float32)).unsqueeze(0),     # (1, y, x)
                    torch.from_numpy(vel_x_target.astype(np.float32)).unsqueeze(0),    # (1, y, x)
                    torch.from_numpy(vel_y_target.astype(np.float32)).unsqueeze(0),    # (1, y, x)
                    torch.from_numpy(c_xx_target.astype(np.float32)).unsqueeze(0),     # (1, y, x)
                    torch.from_numpy(c_xy_target.astype(np.float32)).unsqueeze(0),     # (1, y, x)
                    torch.from_numpy(c_yx_target.astype(np.float32)).unsqueeze(0),     # (1, y, x)
                    torch.from_numpy(c_yy_target.astype(np.float32)).unsqueeze(0),     # (1, y, x)
                ], dim=0)  # (8, y, x)
                
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data to prevent training crashes
            inputs = torch.zeros(self.input_dim, self.resolution, self.resolution)
            labels = torch.zeros(self.input_dim, self.resolution, self.resolution)
        
        # Apply normalization
        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]
        
        # Normalize time
        time_normalized = t / self.constants["time"]
        
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
