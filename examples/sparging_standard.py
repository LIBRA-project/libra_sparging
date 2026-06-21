from sparging import (
    get_sim_input_standard,
    SimulationInput,
    ureg,
    Simulation,
    animation,
)
import logging
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import pint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

FOLDER = Path("paper/standard")
FOLDER.mkdir(exist_ok=True, parents=True)


standard_input = get_sim_input_standard()

standard_input.signal_sparging = lambda t: 0 if t < 8 * ureg.hours else 1
standard_input.signal_irr = lambda t: 1 if t <= 8 * ureg.hours else 0

my_simulation = Simulation(
    standard_input,
    t_final=4 * ureg.days,
    profile_pressure_hydrostatic=True,
)

if __name__ == "__main__":
    output = my_simulation.solve()
    # save output to file
    output.to_json(FOLDER / "standard.json")

    output.profiles_to_csv(FOLDER)

    # # plot results
    # from sparging import plotting
    # plotting.plot_animation(output)

    animation.create_animation(output, show_activity=False)
