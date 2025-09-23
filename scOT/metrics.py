import numpy as np


def lp_error(preds: np.ndarray, targets: np.ndarray, p=1):
    num_samples, num_channels, _, _ = preds.shape
    preds = preds.reshape(num_samples, num_channels, -1)
    targets = targets.reshape(num_samples, num_channels, -1)
    errors = np.sum(np.abs(preds - targets) ** p, axis=-1)
    return np.sum(errors, axis=-1) ** (1 / p)


def relative_lp_error(
    preds: np.ndarray,
    targets: np.ndarray,
    p=1,
    return_percent=True,
):
    num_samples, num_channels, _, _ = preds.shape
    preds = preds.reshape(num_samples, num_channels, -1)
    targets = targets.reshape(num_samples, num_channels, -1)
    errors = np.sum(np.abs(preds - targets) ** p, axis=-1)
    normalization_factor = np.sum(np.abs(targets) ** p, axis=-1)

    # catch 0 division
    normalization_factor = np.sum(normalization_factor, axis=-1)
    normalization_factor = np.where(
        normalization_factor == 0, 1e-10, normalization_factor
    )

    errors = (np.sum(errors, axis=-1) / normalization_factor) ** (1 / p)

    if return_percent:
        errors *= 100

    return errors


def mean_relative_lp_error(
    preds: np.ndarray,
    targets: np.ndarray,
    p=1,
    return_percent=True,
):
    errors = relative_lp_error(preds, targets, p, return_percent)
    return np.mean(errors, axis=0)


def median_relative_lp_error(
    preds: np.ndarray,
    targets: np.ndarray,
    p=1,
    return_percent=True,
):
    errors = relative_lp_error(preds, targets, p, return_percent)
    return np.median(errors, axis=0)


def vrmse(preds: np.ndarray, targets: np.ndarray):
    """
    Calculate VRMSE averaged over channels to report a single number per sample.
    
    This calculates VRMSE for each channel separately, then averages them.
    This is the proper way to get a single VRMSE metric that respects the 
    different physical quantities in each channel.
    
    Args:
        preds: Predicted values with shape (num_samples, num_channels, height, width)
        targets: True values with shape (num_samples, num_channels, height, width)
        
    Returns:
        VRMSE values averaged over channels with shape (num_samples,)
    """
    per_channel_vrmse = vrmse_per_channel(preds, targets)
    
    return np.mean(per_channel_vrmse, axis=1)


def vrmse_per_channel(preds: np.ndarray, targets: np.ndarray):
    """
    Calculate VRMSE for each channel separately.
    
    Args:
        preds: Predicted values with shape (num_samples, num_channels, height, width)
        targets: True values with shape (num_samples, num_channels, height, width)
        
    Returns:
        VRMSE values for each channel with shape (num_samples, num_channels)
    """
    epsilon = 1e-7
    num_samples, num_channels, height, width = preds.shape
    
    vrmse_values = np.zeros((num_samples, num_channels))
    
    for channel in range(num_channels):
        preds_channel = preds[:, channel, :, :].reshape(num_samples, -1)  # (num_samples, height*width)
        targets_channel = targets[:, channel, :, :].reshape(num_samples, -1)  # (num_samples, height*width)
        
        targets_mean = np.mean(targets_channel, axis=1, keepdims=True)  # (num_samples, 1)
        
        mse = np.mean((targets_channel - preds_channel) ** 2, axis=1)  # (num_samples,)
        
        variance = np.mean((targets_channel - targets_mean) ** 2, axis=1)  # (num_samples,)
        
        vrmse_values[:, channel] = np.sqrt(mse / (variance + epsilon))
    
    return vrmse_values
