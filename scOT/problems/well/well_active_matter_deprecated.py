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
        
        # Time constraint - we have 81 actual timesteps per trajectory
        assert self.max_num_time_steps * self.time_step_size <= 81
        
        # Dataset parameters
        self.N_max = 10000  # Large enough to accommodate all data
        self.N_val = 100
        self.N_test = 200
        
        # Data specifications
        self.resolution = 256
        self.input_dim = 11  # 1 (concentration) + 2 (velocity) + 4 (D) + 4 (E)
        self.output_dim = 11  # Same as input
        self.label_description = "concentration,velocity_x,velocity_y,D_00,D_01,D_10,D_11,E_00,E_01,E_10,E_11"
        
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
        
        for field, shape in field_shapes.items():
            if field in stats:
                field_mean = stats[field]['mean']
                field_std = stats[field]['std']
                
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
        timesteps_per_sample = 81 - self.max_num_time_steps * self.time_step_size + 1
        return self.num_trajectories * timesteps_per_sample
    
    def __getitem__(self, idx):
        """Load a single sample."""
        # Map linear index to time-dependent sample
        i, t, t1, t2 = self._idx_map(idx)
        
        # Map to actual sample and time indices
        timesteps_per_sample = 81 - self.max_num_time_steps * self.time_step_size + 1
        sample_idx = i // timesteps_per_sample
        time_offset = i % timesteps_per_sample
        
        # Adjust time indices
        actual_t1 = t1 + time_offset
        actual_t2 = t2 + time_offset
        
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
