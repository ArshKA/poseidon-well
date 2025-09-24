import os
import torch
import numpy as np
import netCDF4
import yaml
import torch.nn.functional as F  # for padding
from ..base import BaseTimeDataset


class WellHelmholtzStaircase(BaseTimeDataset):
    """Well Helmholtz Staircase dataset using assembled NetCDF file."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Time constraint - we have 50 actual timesteps per trajectory (0 to 49)
        assert self.max_num_time_steps * self.time_step_size <= 50
        
        # Dataset parameters
        self.N_max = 10000  # Large enough to accommodate all data
        self.N_val = 100
        self.N_test = 200
        
        # Data specifications
        self.original_resolution = (256, 1024)  # (height, width)
        h0, w0 = self.original_resolution
        self.resolution = max(h0, w0)  # INT side length of the square (e.g., 1024)

        self.input_dim = 2  # pressure_re + pressure_im
        self.output_dim = 2  # Same as input
        self.label_description = "[pressure_re],[pressure_im]"
        
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
                f"Please run: python assemble_helmholtz_staircase.py "
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
                "time": 49.0
            }
        
        with open(stats_file, 'r') as f:
            stats = yaml.safe_load(f)
        
        constants = {
            "time": 49.0,  # Time goes from 0 to 49
        }
        
        field_shapes = {
            'pressure_re': (1,),  # scalar
            'pressure_im': (1,),  # scalar
        }
        
        mean_values, std_values = [], []
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
        # If there are 50 timesteps (0-49), the last input can be at t=49-time_step_size.
        timesteps_per_sample = 50 - self.time_step_size
        return self.num_trajectories * timesteps_per_sample

    @staticmethod
    def _get_square_pad(h, w):
        """
        Compute symmetric padding (left, right, top, bottom) to make (h, w) square.
        Returns: pads tuple for F.pad and the resulting square size.
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
        """Load a single sample corresponding to a linear index."""
        
        # ### --- CORRECTED MAPPING LOGIC --- ###
        timesteps_per_sample = 50 - self.time_step_size
        if timesteps_per_sample <= 0:
            raise ValueError(
                f"time_step_size ({self.time_step_size}) is too large for the "
                f"available 50 timesteps."
            )

        sample_idx = idx // timesteps_per_sample
        time_offset = idx % timesteps_per_sample
        actual_t1 = time_offset
        actual_t2 = time_offset + self.time_step_size

        if actual_t2 > 49:
            raise IndexError(
                f"Calculated target time index {actual_t2} is out of bounds for idx {idx}. "
                f"Max timestep is 49."
            )
        # ### --- END OF CORRECTION --- ###
        
        try:
            with netCDF4.Dataset(self.data_file, 'r') as dataset:
                pressure_re_input = dataset.variables['pressure_re'][sample_idx, actual_t1, :, :]
                pressure_im_input = dataset.variables['pressure_im'][sample_idx, actual_t1, :, :]
                pressure_re_target = dataset.variables['pressure_re'][sample_idx, actual_t2, :, :]
                pressure_im_target = dataset.variables['pressure_im'][sample_idx, actual_t2, :, :]
                
                inputs = torch.stack([
                    torch.from_numpy(pressure_re_input.astype(np.float32)),
                    torch.from_numpy(pressure_im_input.astype(np.float32)),
                ], dim=0)
                
                labels = torch.stack([
                    torch.from_numpy(pressure_re_target.astype(np.float32)),
                    torch.from_numpy(pressure_im_target.astype(np.float32)),
                ], dim=0)
                
        except Exception as e:
            print(f"Error loading sample idx={idx} (sample_idx={sample_idx}, t1={actual_t1}, t2={actual_t2}): {e}")
            side = self.resolution
            inputs = torch.zeros(self.input_dim, side, side, dtype=torch.float32)
            labels = torch.zeros(self.input_dim, side, side, dtype=torch.float32)

            time_normalized = actual_t1 / self.constants["time"]
            return {
                "pixel_values": inputs,
                "labels": labels,
                "time": time_normalized
            }
        
        # Normalize first so padded pixels are exactly 0 afterward
        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]
        
        # Pad to square using zeros in normalized space
        h, w = inputs.shape[-2], inputs.shape[-1]
        if h != w:
            pads, _ = self._get_square_pad(h, w)
            inputs = F.pad(inputs, pads, mode='constant', value=0.0)
            labels = F.pad(labels, pads, mode='constant', value=0.0)
        
        # Normalize time
        time_normalized = actual_t1 / self.constants["time"]
        
        return {
            "pixel_values": inputs,  # (C, S, S) where S = self.resolution
            "labels": labels,        # (C, S, S)
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