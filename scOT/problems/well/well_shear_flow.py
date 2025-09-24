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
        
        # Time constraint - we have 200 actual timesteps per trajectory (0 to 199)
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
                "time": 199.0
            }
        
        with open(stats_file, 'r') as f:
            stats = yaml.safe_load(f)
        
        # Convert to torch tensors with proper shapes for broadcasting
        constants = {
            "time": 199.0,  # Time goes from 0 to 199
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
        # The number of possible starting points for a single time step prediction.
        # If there are 200 timesteps (0-199), the last input can be at t=199-time_step_size.
        timesteps_per_sample = 200 - self.time_step_size
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
        
        # ### --- CORRECTED MAPPING LOGIC --- ###
        timesteps_per_sample = 200 - self.time_step_size
        if timesteps_per_sample <= 0:
            raise ValueError(
                f"time_step_size ({self.time_step_size}) is too large for the "
                f"available 200 timesteps."
            )

        sample_idx = idx // timesteps_per_sample
        time_offset = idx % timesteps_per_sample
        actual_t1 = time_offset
        actual_t2 = time_offset + self.time_step_size

        if actual_t2 > 199:
            raise IndexError(
                f"Calculated target time index {actual_t2} is out of bounds for idx {idx}. "
                f"Max timestep is 199."
            )
        # ### --- END OF CORRECTION --- ###
        
        try:
            with netCDF4.Dataset(self.data_file, 'r') as dataset:
                # Load input fields at time actual_t1
                tracer_input = dataset.variables['tracer'][sample_idx, actual_t1, :, :]
                pressure_input = dataset.variables['pressure'][sample_idx, actual_t1, :, :]
                velocity_input = dataset.variables['velocity'][sample_idx, actual_t1, :, :, :]
                
                # Load target fields at time actual_t2
                tracer_target = dataset.variables['tracer'][sample_idx, actual_t2, :, :]
                pressure_target = dataset.variables['pressure'][sample_idx, actual_t2, :, :]
                velocity_target = dataset.variables['velocity'][sample_idx, actual_t2, :, :, :]
                
                # Convert to tensors and combine
                inputs = torch.cat([
                    torch.from_numpy(tracer_input.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(pressure_input.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(velocity_input.astype(np.float32)).permute(2, 0, 1), # (y, x, 2) -> (2, y, x)
                ], dim=0)
                
                labels = torch.cat([
                    torch.from_numpy(tracer_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(pressure_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(velocity_target.astype(np.float32)).permute(2, 0, 1), # (y, x, 2) -> (2, y, x)
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
        time_normalized = actual_t1 / self.constants["time"]
        
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
    
    def get_ground_truth_for_rollout(self, idx, ar_steps_list):
        """
        Implementation of the ground truth rollout fetching for the 
        WellShearFlow dataset.
        """
        # This logic is moved directly from the old inference script.
        # It correctly maps the linear index to the specific data slice.
        timesteps_per_trajectory = 200 - self.time_step_size
        
        sample_idx = idx // timesteps_per_trajectory
        time_offset = idx % timesteps_per_trajectory
        
        rollout_step_labels = []
        current_time = time_offset
        
        # Open the data file once to fetch all required steps.
        with netCDF4.Dataset(self.data_file, 'r') as nc_dataset:
            for step_size in ar_steps_list:
                target_time = current_time + step_size

                if target_time > 199:
                    raise IndexError(f"Attempted to read from time index {target_time} which is out of bounds for idx {idx}.")

                # Load data for the target time step.
                tracer_target = nc_dataset.variables['tracer'][sample_idx, target_time, :, :]
                pressure_target = nc_dataset.variables['pressure'][sample_idx, target_time, :, :]
                velocity_target = nc_dataset.variables['velocity'][sample_idx, target_time, :, :, :]
                
                # Replicate the exact same preprocessing as in __getitem__.
                label_tensor = torch.cat([
                    torch.from_numpy(tracer_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(pressure_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(velocity_target.astype(np.float32)).permute(2, 0, 1), # (y, x, 2) -> (2, y, x)
                ], dim=0)
                
                # Apply normalization first so padded pixels are exactly 0 afterward
                normalized_label = (label_tensor - self.constants["mean"]) / self.constants["std"]
                
                # Pad to square (channel-wise) using zeros in normalized space
                h, w = normalized_label.shape[-2], normalized_label.shape[-1]
                if h != w:
                    pads, _ = self._get_square_pad(h, w)
                    normalized_label = F.pad(normalized_label, pads, mode='constant', value=0.0)
                
                rollout_step_labels.append(normalized_label)
                
                current_time = target_time
        
        # Return a stacked tensor for this specific initial condition's full rollout.
        return torch.stack(rollout_step_labels, dim=0)