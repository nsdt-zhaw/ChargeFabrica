"""Shared sweep helpers compatible with Python 2.7 and newer."""

import os
import numpy as np

def append_to_npy(output_dir, filename, new_data):
    """Append one result along a new leading axis in a NumPy file."""
    path = os.path.join(output_dir, filename)
    new_data = np.expand_dims(new_data, axis=0)
    if os.path.isfile(path):
        new_data = np.concatenate((np.load(path), new_data), axis=0)
    np.save(path, new_data)

def run_sweep(simulate_device, script_path, sweep_folder,
              start_message, completed_message):
    """Create the output directory and run a simulation sweep."""
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    output_dir = os.path.join(".", "Outputs", script_name, sweep_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(start_message)
    results = simulate_device(output_dir=output_dir)
    print(completed_message)
    return results
