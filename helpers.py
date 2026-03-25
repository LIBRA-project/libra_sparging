import yaml
import numpy as np


# TODO this could be a dataclass?
# see issue #3
def get_input(input_path):
    with open(input_path, "r") as file:
        params = yaml.safe_load(file)
    return params


def setup_yaml_numpy():
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
