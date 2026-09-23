"""Shared sweep helpers compatible with Python 2.7 and newer."""

import os
import glob
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


def prepare_voltage_output(output_dir):
    """Reserve a fresh point directory; never mix separate simulation runs."""
    directory = os.path.join(output_dir, "points")
    if os.path.exists(directory):
        raise ValueError("Results already exist in %s. Choose a new sweep_folder "
                         "or move the previous results before running again." % directory)
    os.mkdir(directory)


def solve_and_save_voltage(solve, output_dir, index, voltage, *args):
    """Publish a worker's result immediately, before the rest of its batch ends."""
    result = solve(voltage, *args)
    path = os.path.join(output_dir, "points", "point_%08d.npz" % index)
    temporary = path + ".tmp"
    with open(temporary, "wb") as destination:
        np.savez(destination, applied_voltage=voltage, sweep_index=index, **result)
    os.rename(temporary, path)
    return result


def load_results(output_dir, keys=None):
    """Load one consistent snapshot in sweep order, or legacy per-field NPY files."""
    directory = os.path.join(output_dir, "points")
    if os.path.isdir(directory):
        paths = sorted(glob.glob(os.path.join(directory, "point_*.npz")))
        if not paths:
            raise ValueError("No completed voltage points yet in %s" % output_dir)
        values = {}
        for path in paths:
            with np.load(path) as point:
                if keys is None:
                    keys = [key for key in point.files
                            if key not in ("applied_voltage", "sweep_index")]
                    keys.append("applied_voltages")
                for key in keys:
                    source = "applied_voltage" if key == "applied_voltages" else key
                    values.setdefault(key, []).append(point[source])
        return dict((key, np.asarray(value)) for key, value in values.items())
    if keys is None:
        keys = [os.path.splitext(os.path.basename(path))[0]
                for path in glob.glob(os.path.join(output_dir, "*.npy"))]
    return dict((key, np.load(os.path.join(output_dir, key + ".npy")))
                for key in keys)
