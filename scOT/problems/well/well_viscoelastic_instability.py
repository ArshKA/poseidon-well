import os
import torch
import numpy as np
import netCDF4
import yaml
from tqdm import tqdm
from ..base import BaseTimeDataset


class WellViscoelasticInstability(BaseTimeDataset):
    """
    Well Viscoelastic Instability dataset that handles variable-length trajectories
    by inferring lengths at runtime if not explicitly provided in the NetCDF file.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Time constraint - we have varying timesteps (20-60) per trajectory, use 60 as max
        assert self.max_num_time_steps * self.time_step_size <= 60
        
        # Dataset parameters
        self.N_max = 10000  # Large enough to accommodate all data
        self.N_val = 100
        self.N_test = 200
        
        # Data specifications
        self.resolution = 512
        self.input_dim = 8
        self.output_dim = 8
        self.label_description = "[pressure],[c_zz],[velocity_x,velocity_y],[C_0,C_1,C_2,C_3]"
        
        # Find assembled data file
        self.data_file = self._find_assembled_file()
        
        # Load or infer trajectory lengths and calculate dataset properties
        self.trajectory_lengths, self.num_trajectories, self.max_len = self._load_trajectory_info()
        
        # Create a map of all valid (sample_idx, time_offset) pairs for fast lookups
        self.sample_map = self._create_sample_map()
        
        # Load normalization constants
        self.constants = self._load_normalization_constants()
        
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

    def _load_trajectory_info(self):
        """
        Load trajectory info. If 'trajectory_lengths' variable is not in the
        NetCDF file, infer the lengths by checking for non-zero data.
        """
        with netCDF4.Dataset(self.data_file, 'r') as dataset:
            try:
                # First, try to load the pre-computed lengths (the fast way)
                lengths = dataset.variables['trajectory_lengths'][:]
                print("Found 'trajectory_lengths' variable in NetCDF file. Using pre-computed lengths.")
            except KeyError:
                # If it fails, infer the lengths at runtime (the slower, robust way)
                print("Warning: 'trajectory_lengths' not found. Inferring lengths at runtime.")
                print("This is a one-time cost per dataset instance.")
                
                num_samples = dataset.dimensions['sample'].size
                max_timesteps = dataset.dimensions['time'].size
                lengths = []
                
                # Use 'pressure' as a representative variable to check for data
                pressure_var = dataset.variables['pressure']

                for i in tqdm(range(num_samples), desc="Inferring trajectory lengths"):
                    traj_len = 0
                    # Iterate backwards from the end to find the last valid time step
                    for t in range(max_timesteps - 1, -1, -1):
                        # Check if there is any non-zero value in this time slice
                        if np.any(pressure_var[i, t, :, :]):
                            traj_len = t + 1  # Length is the last index + 1
                            break  # Found the last frame, move to next sample
                    lengths.append(traj_len)
                
                lengths = np.array(lengths)

            num_trajectories = len(lengths)
            max_len = int(np.max(lengths)) if num_trajectories > 0 else 0
            return lengths, num_trajectories, max_len

    def _create_sample_map(self):
        """Create a mapping from a linear index to a (sample_idx, time_offset) pair."""
        sample_map = []
        for i in range(self.num_trajectories):
            # A valid sample is (t, t + time_step_size).
            # The last possible start time 't' is length - 1 - time_step_size.
            num_valid_steps = self.trajectory_lengths[i] - self.time_step_size
            for t in range(num_valid_steps):
                sample_map.append((i, t))
        return sample_map

    def _load_normalization_constants(self):
        """Load normalization constants from stats.yaml."""
        stats_file = os.path.join(self.data_path, "stats.yaml")
        
        # Use the actual maximum time step found in the data for normalization
        max_time = float(self.max_len - 1) if self.max_len > 0 else 59.0

        if not os.path.exists(stats_file):
            print(f"Warning: stats.yaml not found at {stats_file}, using default normalization")
            return {
                "mean": torch.zeros(self.input_dim, 1, 1),
                "std": torch.ones(self.input_dim, 1, 1),
                "time": max_time,
            }
        
        with open(stats_file, 'r') as f:
            stats = yaml.safe_load(f)
        
        constants = {"time": max_time}
        
        means = stats.get('mean', {})
        stds = stats.get('std', {})
        
        mean_values, std_values = [], []

        def _get_stat(stats_dict, key, default, length):
            val = stats_dict.get(key, default)
            if isinstance(val, (int, float)):
                return [float(val)] * length
            flat_val = np.array(val, dtype=np.float32).flatten().tolist()
            return (flat_val + [float(default)] * length)[:length]

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
    
    def __len__(self):
        """Return the total number of valid time-dependent samples across all trajectories."""
        return len(self.sample_map)
    
    def __getitem__(self, idx):
        """Load a single sample using the pre-computed sample map."""
        if idx >= len(self.sample_map):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self.sample_map)}")

        # Look up the correct sample and time from our map
        sample_idx, time_offset = self.sample_map[idx]
        
        actual_t1 = time_offset
        actual_t2 = time_offset + self.time_step_size

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
                    torch.from_numpy(velocity_input.astype(np.float32)).permute(2, 0, 1),
                    torch.from_numpy(C_input.astype(np.float32)).permute(2, 0, 1),
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
            # Return dummy data to prevent training crashes on isolated errors
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
        """
        return self.constants
    
    def get_ground_truth_for_rollout(self, idx, ar_steps_list):
        """
        Implementation of the ground truth rollout fetching, aware of variable trajectory lengths.
        """
        # Map the linear dataset index back to the specific trajectory and start time.
        sample_idx, time_offset = self.sample_map[idx]
        
        # Get the specific length of this trajectory to use for bounds checking.
        trajectory_len = self.trajectory_lengths[sample_idx]
        
        rollout_step_labels = []
        current_time = time_offset
        
        with netCDF4.Dataset(self.data_file, 'r') as nc_dataset:
            for step_size in ar_steps_list:
                target_time = current_time + step_size

                # Check bounds against the actual length of the current trajectory
                if target_time >= trajectory_len:
                    raise IndexError(
                        f"Attempted to read from time index {target_time} which is out of bounds "
                        f"for trajectory {sample_idx} with length {trajectory_len}."
                    )

                # Load data for the target time step.
                pressure_target = nc_dataset.variables['pressure'][sample_idx, target_time, :, :]
                c_zz_target = nc_dataset.variables['c_zz'][sample_idx, target_time, :, :]
                velocity_target = nc_dataset.variables['velocity'][sample_idx, target_time, :, :, :]
                C_target = nc_dataset.variables['C'][sample_idx, target_time, :, :, :]
                
                label_tensor = torch.cat([
                    torch.from_numpy(pressure_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(c_zz_target.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(velocity_target.astype(np.float32)).permute(2, 0, 1),
                    torch.from_numpy(C_target.astype(np.float32)).permute(2, 0, 1),
                ], dim=0)
                
                normalized_label = (label_tensor - self.constants["mean"]) / self.constants["std"]
                rollout_step_labels.append(normalized_label)
                
                current_time = target_time
        
        return torch.stack(rollout_step_labels, dim=0)