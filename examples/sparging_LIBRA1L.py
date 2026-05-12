from sparging.config import ureg
from sparging import all_correlations
from sparging import animation
from sparging.model import Simulation
from sparging.inputs import (
    get_sim_input_LIBRA1L,
)
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

my_input, t_irr = get_sim_input_LIBRA1L()

print(f"t_irr = {t_irr.to('hours')}")
my_input.signal_sparging = lambda t: 0 if t <= t_irr else 1

my_simulation = Simulation(
    my_input,
    t_final=2 * ureg.days,
    profile_pressure_hydrostatic=False,
)

if __name__ == "__main__":
    output = my_simulation.solve(fast_solve=True)

    # # save output to file
    # output.profiles_to_csv(f"output_{tank_height}m.csv")

    # # plot results
    # from sparging import plotting
    # plotting.plot_animation(output)

    animation.create_animation(output, show_activity=False)
