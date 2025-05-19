from scipy.ndimage import gaussian_filter

def smooth(data, sigma):
    """
    Smooth the data using a Gaussian filter.

    Args:
        data: The input array to smooth.
        sigma: The standard deviation for the Gaussian kernel, defines smoothing level.

    Returns:
        The smoothed array.
    """
    smoothed_data = gaussian_filter(data, sigma=sigma)
    return smoothed_data