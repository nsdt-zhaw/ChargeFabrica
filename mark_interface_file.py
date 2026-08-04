from scipy.ndimage import binary_dilation

def _dilate(mask, iterations):
    """Return ``mask`` after the requested number of dilation steps."""
    for unused_step in range(iterations):
        mask = binary_dilation(mask)
    return mask

def mark_interfaces(device_architecture, layer_value_1, layer_value_2,
                    dilation_size=3):
    """Mark the part of layer 2 reached by dilation from layer 1."""
    layer_1 = _dilate(device_architecture == layer_value_1, dilation_size)
    layer_2 = device_architecture == layer_value_2
    return (layer_1 & layer_2).astype(float)

def mark_interfaces_mixed(device_architecture, layer_value_1, layer_value_2,
                          dilation_size=3):
    """Mark overlap after alternately dilating layer 1 and layer 2."""
    layer_1 = device_architecture == layer_value_1
    layer_2 = device_architecture == layer_value_2

    for step in range(dilation_size):
        if step % 2:
            layer_2 = binary_dilation(layer_2)
        else:
            layer_1 = binary_dilation(layer_1)

    return (layer_1 & layer_2).astype(float)