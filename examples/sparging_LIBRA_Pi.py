from sparging import (
    LIBRA_PI_GEOM,
    LIBRA_PI_MAT,
    LIBRA_PI_OPERATING_PARAMS,
    LIBRA_PI_SPARGING_PARAMS,
    SimulationInput,
    ureg,
    Simulation,
    animation,
)
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

LIBRA_PI_OPERATING_PARAMS.temperature = 700 * ureg.celsius

my_input = SimulationInput.from_parameters(
    LIBRA_PI_GEOM,
    LIBRA_PI_MAT,
    LIBRA_PI_OPERATING_PARAMS,
    LIBRA_PI_SPARGING_PARAMS,
)

my_input.signal_sparging = lambda t: 0 if t < 8 * ureg.hours else 1
my_input.signal_irr = lambda t: 1 if t <= 8 * ureg.hours else 0

my_simulation = Simulation(
    my_input,
    t_final=4 * ureg.days,
)

if __name__ == "__main__":
    output = my_simulation.solve()
    # output.to_json("libra_pi_output_700C.json")

    # # save output to file
    # output.profiles_to_csv(f"output_{tank_height}m.csv")

    # # plot results
    # from sparging import plotting
    # plotting.plot_animation(output)

    animation.create_animation(output, show_activity=False)
