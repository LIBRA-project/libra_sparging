from __future__ import annotations
from sparging.config import ureg
from sparging import all_correlations
from sparging import animation
from sparging.model import Simulation
from sparging.inputs import (
    get_sim_input_standard,
)
import sparging.postprocess as pp
import logging
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import pint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


my_input = get_sim_input_standard()
# my_input = get_sim_input(librapi_input_dict)

print(my_input.get_S_T())
print(f"{my_input.Q_T.to('molT/s')} = {my_input.Q_T.to('molT2/hour')}")
print(my_input.volume.to("m^3"))
print(
    f"Concentration at steady state (no dispersion, no PP limited): {
        my_input.get_c_T2_SS().to('molT2/m^3')
    }"
)
tau = my_input.get_tau().to("seconds")
print(f"tau = {tau.to('hours')}")
print(f"Partial pressure number PP = {my_input.get_PP_number()}")


def profile_source_T(z: pint.Quantity | list[float], height: pint.Quantity = None):
    import numpy as np

    if isinstance(z, (float, np.ndarray, list)):  # non-dimensional height (0 to 1)
        # return np.pi / 2 * np.sin(np.pi * z)  # normalized
        return 1 + 1 * np.sin(np.pi * z)  # not normalized
    if isinstance(z, ureg.Quantity):
        assert False
        if height is None:
            raise ValueError("Must provide height if z is a dimensional quantity")
        return np.pi / 2 * np.sin(np.pi / height * z)  # normalized
    else:
        raise NotImplementedError("z must be either a float or a pint.Quantity")


t_final = 50 * ureg.hour
t_irr = 20 * ureg.hour
my_input.signal_irr = lambda t: 1 if t <= t_irr else 0
my_input.signal_sparging = lambda t: 1

my_simulation = Simulation(
    my_input,
    t_final=t_final,
    profile_pressure_hydrostatic=True,
)

if __name__ == "__main__":
    # my_simulation.sim_input.E_g *= 1e5
    # my_simulation.sim_input.E_l *= 1e-5
    output = my_simulation.solve(fast_solve=True)
    output.to_json("temp.json")
    popt, pcov = pp.fit_exp(
        output.inventories_T2_salt, output.times, 0 * ureg.s, t_irr, phase="rampup"
    )
    print(f"Fitted parameters: n0 = {popt[1]}, tau = {popt[0].to('h')}")
    popt, pcov = pp.fit_exp(
        output.inventories_T2_salt, output.times, t_irr, t_final, phase="decay"
    )
    print(f"Fitted parameters: n0 = {popt[1]}, tau = {popt[0].to('h')}")
    animation.create_animation(output, show_activity=False)
