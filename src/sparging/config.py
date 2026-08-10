import subprocess
import yaml
import numpy as np
from pint import UnitRegistry
import scipy.constants as const
import logging

"""
Naming convention (paper symbol <-> code)
=========================================
Core rule:  <quantity>_<species>  mirrors the paper symbol, e.g.
    c_T2   -> c_{T2}      ndot_T2 -> \\dot n_{T2}     P_g/P_l -> P_g/P_l

Collections are named by the AXIS they vary along:
    _profile   : varies in space z           (1D array)
    _profiles  : space x time                (2D array, leading axis = time)
    _series    : varies in time              (1D array)

Fixed suffix vocabulary:
    Xdot    : time derivative / rate   (\\dot X)   -> ndot_T2, Vdot_g0
    X_ave   : spatial/quantity average (\\bar X)   -> gen_T2_ave
    X_0     : tank bottom / gas inlet  
    X_init  : initial condition (t=0) 
"""

molar_mass_T2 = 3.016 * 2  # g/mol T2
specific_activity_tritium = 3.57e14  # Bq/g
molT2_to_activity = molar_mass_T2 * specific_activity_tritium  # Bq/mol T2

ureg = UnitRegistry(
    autoconvert_offset_to_baseunit=True
)  # to deal with offset units (eg: degree celsius)
ureg.setup_matplotlib(True)
ureg.formatter.default_format = ".3e~D"
ureg.define("triton = [tritium] = T")
ureg.define(f"molT = {const.N_A} * triton")
ureg.define(f"molT2 = 2 * {const.N_A} * triton")
ureg.define("neutron = [neutron] = n")
ureg.define("sccm = 7.44e-7 mol/s")  # holds for an ideal gas
ureg.define(f"Bq = {1 / molT2_to_activity} * molT2")


const_R = const.R * ureg("J/K/mol")  # ideal gas constant
const_g = const.g * ureg("m/s**2")  # gravitational acceleration

VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")


def verbose(self, message, *args, **kws):
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, message, args, **kws)


logging.Logger.verbose = verbose


# --- provenance and I/O helpers ---------------------------------------------


def get_input(yaml_input_path) -> dict:
    """Load a YAML input file, unwrapping a top level `input:` key if present."""
    with open(yaml_input_path, "r") as file:
        params = yaml.safe_load(file)
        if "input" in params:
            params = params["input"]
    return params


def setup_yaml():
    """Tell PyYAML to represent numpy scalars in a human readable way."""

    def numpy_representer(dumper, data):
        return dumper.represent_data(data.item())

    yaml.add_representer(np.float64, numpy_representer)


def get_git_hash() -> str:
    """Short commit hash of the working tree, suffixed `-dirty` when it has
    uncommitted changes. Datasets record this, so the suffix is what tells a
    reader whether a run is actually reproducible from that commit."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode != 0
    return f"{commit}-dirty" if dirty else commit
