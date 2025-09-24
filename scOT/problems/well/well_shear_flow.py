import os
import torch
import numpy as np
import netCDF4
import yaml
import torch.nn.functional as F  # for padding
from ..base import BaseTimeDataset


class WellShearFlow(BaseTimeDataset):
    """Well Shear Flow dataset using assembled NetCDF file."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Time constraint - we have 200 actual timesteps per trajectory
        assert self.max_num_time_steps * self.time_step_size <= 200
        
        # Dataset parameters
        self.N_max = 10000  # Large enough to accommodate all data
        self.N_val = 100
        self.N_test = 200
        
        # Data specifications
        self.original_resolution = (256, 512)  # (height, width)
        h0, w0 = self.original_resolution
        self.resolution = max(h0, w0)  # INT side length (e.g., 512)

        self.input_dim = 4  # tracer + pressure + velocity_x + velocity_y
        self.output_dim = 4  # Same as input
        self.label_description = "[tracer],[pressure],[velocity_x],[velocity_y]"
        
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
                f"Please run: python assemble_shear_flow.py "
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
                "time": 200.0
            }
        
        with open(stats_file, 'r') as f:
            stats = yaml.safe_load(f)
        
        # Convert to torch tensors with proper shapes for broadcasting
        constants = {
            "time": 200.0,  # Time goes from 0 to 200
        }
        
        # Process each field's normalization
        field_shapes = {
            'tracer': (1,),      # scalar
            'pressure': (1,),    # scalar
            'velocity': (2,),    # vector [x, y]
        }
        
        mean_values = []
        std_values = []
        
        means = stats.get('mean', {})
        stds = stats.get('std', {})
        
        for field, shape in field_shapes.items():
            if field in means and field in stds:
                field_mean = means[field]
                field_std = stds[field]
                
                if isinstance(field_mean, (int, float)):
                    mean_values.extend([field_mean] * shape[0])
                    std_values.extend([field_std] * shape[0])
                else:
                    mean_values.extend(field_mean)
                    std_values.extend(field_std)
            else:
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
        # Each sample can provide (n_timesteps - max_num_time_steps * time_step_size + 1) time samples
        timesteps_per_sample = 200 - self.max_num_time_steps * self.time_step_size + 1
        return self.num_trajectories * timesteps_per_sample

    @staticmethod
    def _get_square_pad(h, w):
        """
        Compute symmetric padding (left, right, top, bottom) to make (h, w) square.
        Returns a pads tuple for F.pad and the resulting square size.
        """
        size = max(h, w)
        pad_w = size - w
        pad_h = size - h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return (left, right, top, bottom), size

    def remove_padding(self, tensor, original_hw=None):
        """
        Remove the square padding and recover the original (non-square) field.
        Works for tensors shaped (..., H, W) or (C, H, W) or (B, C, H, W).
        
        Args:
            tensor: padded tensor
            original_hw: (h, w) to crop back to. Defaults to dataset's original_resolution.
        """
        if original_hw is None:
            original_hw = self.original_resolution
        h, w = original_hw
        pads, _ = self._get_square_pad(h, w)
        left, right, top, bottom = pads
        slicer = (..., slice(top, top + h), slice(left, left + w))
        return tensor[slicer]
    
    def __getitem__(self, idx):
        """Load a single sample."""
        # Map linear index to time-dependent sample
        i, t, t1, t2 = self._idx_map(idx)
        
        # Map to actual sample and time indices
        timesteps_per_sample = 200 - self.max_num_time_steps * self.time_step_size + 1
        sample_idx = i % self.num_trajectories  # Which trajectory
        time_offset = i // self.num_trajectories  # Which time window within trajectory
        
        # Adjust time indices
        actual_t1 = t1 + time_offset
        actual_t2 = t2 + time_offset
        
        try:
            with netCDF4.Dataset(self.data_file, 'r') as dataset:
                # Load input fields at time actual_t1 (already (y, x) in file)
                tracer_input = dataset.variables['tracer'][sample_idx, actual_t1, :, :]        # (y, x)
                pressure_input = dataset.variables['pressure'][sample_idx, actual_t1, :, :]    # (y, x)
                velocity_input = dataset.variables['velocity'][sample_idx, actual_t1, :, :, :] # (y, x, 2)
                vel_x_input = velocity_input[:, :, 0]  # (y, x)
                vel_y_input = velocity_input[:, :, 1]  # (y, x)
                
                # Load target fields at time actual_t2
                tracer_target = dataset.variables['tracer'][sample_idx, actual_t2, :, :]        # (y, x)
                pressure_target = dataset.variables['pressure'][sample_idx, actual_t2, :, :]    # (y, x)
                velocity_target = dataset.variables['velocity'][sample_idx, actual_t2, :, :, :] # (y, x, 2)
                vel_x_target = velocity_target[:, :, 0]  # (y, x)
                vel_y_target = velocity_target[:, :, 1]  # (y, x)
                
                # Stack and convert to (4, y, x)
                inputs = torch.cat([
                    torch.from_numpy(tracer_input.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(pressure_input.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(vel_x_input.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(vel_y_input.astype(np.float32)).unsqueeze(0),
                ], dim=0)
                
                labels = torch.cat([
                    torch.from_numpy(tracer_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(pressure_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(vel_x_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(vel_y_target.astype(np.float32)).unsqueeze(0),
                ], dim=0)
                
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data to prevent training crashes; match the square shape
            side = self.resolution
            inputs = torch.zeros(self.input_dim, side, side, dtype=torch.float32)
            labels = torch.zeros(self.input_dim, side, side, dtype=torch.float32)
            # Normalize time and return early (next normalization is no-op for zeros)
            inputs = (inputs - self.constants["mean"]) / self.constants["std"]
            labels = (labels - self.constants["mean"]) / self.constants["std"]
            time_normalized = t / self.constants["time"]
            return {
                "pixel_values": inputs,
                "labels": labels,
                "time": time_normalized
            }
        
        # Apply normalization first so padded pixels are exactly 0 afterward
        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]
        
        # Pad to square (channel-wise) using zeros in normalized space
        h, w = inputs.shape[-2], inputs.shape[-1]
        if h != w:
            pads, _ = self._get_square_pad(h, w)
            inputs = F.pad(inputs, pads, mode='constant', value=0.0)
            labels = F.pad(labels, pads, mode='constant', value=0.0)
        
        # Normalize time
        time_normalized = t / self.constants["time"]
        
        return {
            "pixel_values": inputs,  # shape: (C, S, S) where S = self.resolution
            "labels": labels,        # shape: (C, S, S)
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
