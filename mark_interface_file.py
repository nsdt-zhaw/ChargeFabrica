import numpy as np
from scipy.ndimage import binary_dilation

#Places the interface inside layer_value_2
def mark_interfaces_backup(device_architecture, layer_value_1, layer_value_2, dilation_size=3):
    """
    Marks the interfacial region between two layers in a device architecture of any dimension.

    Arguments:
    - device_architecture: n-dimensional array representing the device architecture.
    - layer_value_1: the value in the device_architecture array that represents the first layer.
    - layer_value_2: the value in the device_architecture array that represents the second layer.
    - dilation_size: how much to dilate the boundary of the layers. Default is 3.

    Returns an array with same shape as device_architecture with the interface marked, where the value is 1.
    """
    # Find locations of each layer
    location_layer_1 = np.where(device_architecture == layer_value_1, 1, 0)
    location_layer_2 = np.where(device_architecture == layer_value_2, 1, 0)

    # Create a symmetric structuring element
    if dilation_size % 2 == 0:  # Even size
        adjusted_size = dilation_size + 1  # Convert to odd
    else:
        adjusted_size = dilation_size

    # Dilate the boundary of the first layer
    structure = np.ones([adjusted_size] * device_architecture.ndim)
    dilated_location_layer_1 = binary_dilation(location_layer_1, structure=structure)

    # Apply Boolean AND to get the common regions between dilated first layer and second layer
    location_interface = dilated_location_layer_1 & location_layer_2

    return location_interface.astype(float)

def mark_interfaces(device_architecture, layer_value_1, layer_value_2, dilation_size=3):
    """
    Marks the interfacial region between two layers in a device architecture of any dimension.

    Arguments:
    - device_architecture: n-dimensional array representing the device architecture.
    - layer_value_1: the value in the device_architecture array that represents the first layer.
    - layer_value_2: the value in the device_architecture array that represents the second layer.
    - dilation_size: how much to dilate the boundary of the layers. Default is 3.

    Returns an array with same shape as device_architecture with the interface marked, where the value is 1.
    """
    # Find locations of the two layers and convert them to boolean arrays
    location_layer_1 = np.where(device_architecture == layer_value_1, True, False)
    location_layer_2 = np.where(device_architecture == layer_value_2, True, False)
    # Initialize the interfacial region as a boolean array
    interface_region = np.zeros_like(device_architecture, dtype=bool)
    # Iteratively dilate location_layer_1
    for step in range(dilation_size):
        location_layer_1 = binary_dilation(location_layer_1)
        interface_region |= location_layer_1 & location_layer_2  # Add overlap to interface

    return interface_region.astype(float)

def mark_interfaces_mixed(device_architecture, layer_value_1, layer_value_2, dilation_size=3):
    """
    Marks the interfacial region symmetrically between two layers in a device architecture,
    by iterating dilation alternately on both sides for a given number of dilation steps.

    Arguments:
    - device_architecture: n-dimensional array representing the device architecture.
    - layer_value_1: the value in the device_architecture array that represents the first layer.
    - layer_value_2: the value in the device_architecture array that represents the second layer.
    - dilation_size: The number of dilation steps to apply. Default is 3.

    Returns an array with the same shape as device_architecture with the interface marked, where the value is 1.
    """
    # Find locations of the two layers and convert them to boolean arrays
    location_layer_1 = np.where(device_architecture == layer_value_1, True, False)
    location_layer_2 = np.where(device_architecture == layer_value_2, True, False)

    # Create a symmetric structuring element
    if dilation_size % 2 == 0:  # Even size
        adjusted_size = 2
    else:
        adjusted_size = 2

    # Dilate the boundary of the first layer
    structuring_element = np.ones([adjusted_size] * device_architecture.ndim)

    # Initialize the interfacial region as a boolean array
    interface_region = np.zeros_like(device_architecture, dtype=bool)

    # Iteratively dilate each layer, alternating between them and combining overlap
    for step in range(dilation_size):
        if step % 2 == 0:
            # Dilate layer 1 and compare with layer 2
            location_layer_1 = binary_dilation(location_layer_1)
            interface_region |= location_layer_1 & location_layer_2  # Add overlap to interface
        if step % 2 == 1:
            # Dilate layer 2 and compare with layer 1
            location_layer_2 = binary_dilation(location_layer_2)
            interface_region |= location_layer_1 & location_layer_2  # Add overlap to interface

    # Return the interface as a float array (1.0 where interface exists, 0.0 elsewhere)
    return interface_region.astype(float)

