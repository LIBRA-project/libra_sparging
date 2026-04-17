from sparging.config import ureg
from sparging import all_correlations
from sparging import animation
from sparging.model import Simulation
from sparging.inputs import (
    ColumnGeometry,
    BreederMaterial,
    OperatingParameters,
    SpargingParameters,
    SimulationInput,
)
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


geom = ColumnGeometry(
    area=0.2 * ureg.m**2,
    height=1.0 * ureg.m,
    nozzle_diameter=0.001 * ureg.m,
    nb_nozzle=10 * ureg.dimensionless,
)

flibe = BreederMaterial(
    name="FLiBe",
)

operating_params = OperatingParameters(
    temperature=600 * ureg.celsius,
    P_top=1 * ureg.atm,
    flow_g_mol=400 * ureg.sccm,
    tbr=0.1 * ureg("triton / neutron"),
    n_gen_rate=1e9 * ureg("neutron / s"),
)

sparging_params = SpargingParameters(
    h_l=all_correlations("h_l_briggs"),
)


# class method from_parameters that takes in objects like ColumnGeometry, BreederMaterial, OperatingParameters and returns a SimulationInput object with the appropriate correlations for the given parameters. This method should be able to handle cases where some of the parameters are already provided as correlations and should not overwrite them.
my_input = SimulationInput.from_parameters(
    geom, flibe, operating_params, sparging_params
)
logger.info(my_input)


def profile_source_T(z: pint.Quantity):
    import numpy as np

    # return np.sin(np.pi / (1 * ureg.m) * z)
    return 0.5 * (1 + np.cos(0.5 * np.pi / (1 * ureg.m) * z))


my_simulation = Simulation(
    my_input,
    t_final=3 * ureg.days,
    signal_irr=lambda t: 1 if t < 12 * ureg.hour else 0,
    signal_sparging=lambda t: 1,
)
output = my_simulation.solve()

# # save output to file
# output.profiles_to_csv(f"output_{tank_height}m.csv")

# # plot results
# from sparging import plotting
# plotting.plot_animation(output)


animation.create_animation(output, show_activity=True)
