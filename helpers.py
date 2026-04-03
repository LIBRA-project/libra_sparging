import yaml
import numpy as np
from config import *


# TODO this could be a dataclass?
# see issue #3
def get_input(yaml_input_path) -> dict:
    with open(yaml_input_path, "r") as file:
        params = yaml.safe_load(file)
        if "input" in params:
            params = params["input"]
    return params


def setup_yaml():
    """Tells PyYAML to represent numpy types in human readable way"""

    def numpy_representer(dumper, data):
        # Convert numpy scalar to a standard Python type
        return dumper.represent_data(data.item())

    # Register the representer for used numpy types (can add other types of needed)
    yaml.add_representer(np.float64, numpy_representer)


def get_git_hash():
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "no-git"


def string_to_ramp(t, start, end):
    (t, start, end) = to_comparable_magnitude([t, start, end])
    return np.clip((t - start) / (end - start), 0, 1)


def string_to_step(t, start):
    (t, start) = to_comparable_magnitude([t, start])
    return np.heaviside(t - start, 1)


def to_comparable_magnitude(quantities: list):
    quantities = [ureg.Quantity(q) for q in quantities]
    base_units = [q.to_base_units().units for q in quantities]
    if len(set(base_units)) > 1:
        raise ValueError(f"Quantities have different base units: {base_units}")
    return tuple([q.to_base_units().magnitude for q in quantities])
