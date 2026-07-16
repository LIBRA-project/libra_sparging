from sparging import (
    get_sim_input_standard,
    get_sim_input_LIBRA_Pi,
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

FOLDER = Path("paper/runs/reference")
FOLDER.mkdir(exist_ok=True, parents=True)


# standard_input = get_sim_input_standard()
my_input = get_sim_input_LIBRA_Pi()


print(f"Pi = {my_input.get_Pi_number():.2f}")
print(f"steady state c_T2 = {my_input.get_c_T2_SS():.2e}")
print(f"Bo = {my_input.get_Bo():.2e}")

my_input.c_T2_init = 3e-11 * ureg.molT2 / ureg.m**3


my_simulation = Simulation(
    my_input,
    t_final=3 * ureg.days,
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
        "J_T2",
        "eps_g",
        "u_g",
    ]

    tau_pred = my_input.get_tau()
    dt = (tau_pred / 200).to("s")
    dx = (2 * my_input.height / (my_input.get_Bo())).to("m")

    print(f"dx={dx:~.2e}, dt={dt:~.2e}")
    output = my_simulation.solve(dt=dt, dx=dx)

    # save output to file
    output.exports_to_csv(FOLDER)
    output.to_json(
        FOLDER / "summary.json",
        ["analytical_quantities", "fit_summary", "intermediate_params"],
    )

    animation.create_animation(output, show_activity=False)
