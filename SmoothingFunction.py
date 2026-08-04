from scipy.ndimage import gaussian_filter

def flatten_and_smooth_all(arrays, smooth_factor):
    """Optionally smooth each array, then return its flattened form."""
    if smooth_factor > 0.0:
        return [gaussian_filter(array, sigma=smooth_factor).flatten()
                for array in arrays]
    return [array.flatten() for array in arrays]