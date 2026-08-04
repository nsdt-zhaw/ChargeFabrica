"""Functions shared by the one-dimensional plotting scripts."""

import numpy as np


def median_filter_1d(values, kernel_size):
    """Apply a median filter, extending boundaries with endpoint values."""
    assert kernel_size % 2 == 1, "Median filter length must be odd."
    assert values.ndim == 1, "Input must be one-dimensional."
    half_width = (kernel_size - 1) // 2
    window = np.zeros((len(values), kernel_size), dtype=values.dtype)
    window[:, half_width] = values

    for offset in range(half_width):
        distance = half_width - offset
        window[distance:, offset] = values[:-distance]
        window[:distance, offset] = values[0]
        window[:-distance, -(offset + 1)] = values[distance:]
        window[-distance:, -(offset + 1)] = values[-1]

    return np.median(window, axis=1)
