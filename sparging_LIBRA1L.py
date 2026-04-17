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
    area=170 * ureg.cm**2,
    height=7 * ureg.cm,
    nozzle_diameter=1.4 * ureg.mm,
    nb_nozzle=1 * ureg.dimensionless,
)

flibe = BreederMaterial(
    name="FLiBe",
)

operating_params = OperatingParameters(
    temperature=600 * ureg.celsius,
    P_top=1 * ureg.atm,
    flow_g_mol=40 * ureg.sccm,
    tbr=2e-3 * ureg("triton / neutron"),
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
print(my_input.get_S_T())
print(my_input.Q_T.to("molT/s"))

n_fluence = 2.5e13 * ureg("neutron")
n_gen_rate = operating_params.n_gen_rate
t_irr = n_fluence / n_gen_rate
print(f"t_irr = {t_irr.to('seconds')}")
my_simulation = Simulation(
    my_input,
    t_final=50 * ureg.days,
    signal_irr=lambda t: 1 if t < t_irr else 0,
    signal_sparging=lambda t: 0 if t < t_irr else 1,
    # signal_sparging=lambda t: 0,
    profile_pressure_hydrostatic=False,
    profile_source_T=lambda z: 1,
)

if __name__ == "__main__":
    output = my_simulation.solve()

    # # save output to file
    # output.profiles_to_csv(f"output_{tank_height}m.csv")

    # # plot results
    # from sparging import plotting
    # plotting.plot_animation(output)

    animation.create_animation(output, show_activity=False)
