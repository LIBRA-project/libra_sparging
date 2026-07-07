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

FOLDER = Path("paper/runs/standard")
FOLDER.mkdir(exist_ok=True, parents=True)


standard_input = get_sim_input_standard()

print(f"Pi = {standard_input.get_Pi_number():.2f}")
print(f"steady state c_T2 = {standard_input.get_c_T2_SS():.2e}")
print(f"Bo = {standard_input.get_Bo():.2e}")
standard_input.test_eps_g()

# standard_input.signal_sparging = lambda t: 0 if t < 24 * ureg.hours else 1
standard_input.signal_sparging = lambda t: 1
# standard_input.signal_irr = lambda t: 1 if t <= 24 * ureg.hours else 0
standard_input.signal_irr = lambda t: 1
# standard_input.profile_source_T = lambda z: 1 - z * ureg.m / standard_input.height
standard_input.c_T2_init = 3e-11 * ureg.molT2 / ureg.m**3

my_simulation = Simulation(
    standard_input,
    t_final=6 * ureg.days,
    dispersion_on=True,
    constant_profiles=False,
)

if __name__ == "__main__":
    my_simulation.exports = [
        "P_g",
        "a",
        "aJ_T2",
        "c_T2",
        "y_T2",
        "P_T2",
        "n_T2_salt",
        "ndot_T2",
    ]
    output = my_simulation.solve(fast_solve=True)

    # save output to file
    output.exports_to_csv(FOLDER)
    output.to_json(FOLDER / "params.json")
    standard_input.to_json(
        FOLDER / "intermediate_params.json"
    )  # inspect intermediate parameters

    animation.create_animation(output, show_activity=False)
