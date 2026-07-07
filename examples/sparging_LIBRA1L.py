from sparging import get_sim_input_LIBRA1L, Simulation, ureg, animation
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
    constant_profiles=True,
)

if __name__ == "__main__":
    output = my_simulation.solve(fast_solve=True)

    # # save output to file
    # output.profiles_to_csv(f"output_{tank_height}m.csv")

    # # plot results
    # from sparging import plotting
    # plotting.plot_animation(output)

    animation.create_animation(output, show_activity=True)
