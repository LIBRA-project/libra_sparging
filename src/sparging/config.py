from pint import UnitRegistry
import scipy.constants as const

VERBOSE = False

ureg = UnitRegistry(
    autoconvert_offset_to_baseunit=True
)  # to deal with offset units (eg: degree celsius)
ureg.formatter.default_format = ".3e~D"
ureg.define("triton = [tritium] = T")
ureg.define(f"molT = {const.N_A} * triton")
ureg.define(f"molT2 = 2 * {const.N_A} * triton")
ureg.define("neutron = [neutron] = n")
ureg.define("sccm = 7.44e-7 mol/s")


const_R = const.R * ureg("J/K/mol")  # ideal gas constant
const_g = const.g * ureg("m/s**2")  # gravitational acceleration
