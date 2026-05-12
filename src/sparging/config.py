from pint import UnitRegistry
import scipy.constants as const
import logging

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
