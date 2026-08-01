from sparging import (
    get_sim_input_LIBRA_Pi,
    ureg,
    Simulation,
    animation,
)
import logging
import time
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
# my_func = my_input.h_l
# my_input.h_l = lambda z: my_func(z) * 9.4


print(f"Pi = {my_input.get_Pi_ave():.2f}")
print(f"steady state c_T2 = {my_input.get_c_T2_SS():.2e}")
print(f"Bo = {my_input.get_Bo():.2e}")

tau = my_input.get_tau()
print(f"tau = {tau:~.2e}")

my_input.c_T2_init = 3e-11 * ureg.molT2 / ureg.m**3


my_simulation = Simulation(
    my_input,
    t_final=5 * tau,
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
        "h_l",
    ]

    tau_pred = my_input.get_tau()
    dt = (tau_pred * 0.005).to(
        "s"
    )  # 2% of tau_pred gives 1% error on tau compared to fine mesh
    dx = my_input.dx_from_Pe(0.1)  # grid Pe = 2
    my_input.K_s *= 0.36
    print(f"dx={dx:~.2e}, dt={dt:~.2e}")
    t_start = time.perf_counter()
    output = my_simulation.solve(dt=dt, dx=dx, verbose=False)
    elapsed = time.perf_counter() - t_start
    print(f"solve() took {elapsed:.1f} s")

    # save output to file
    output.exports_to_csv(FOLDER)
    output.to_json(
        FOLDER / "summary.json",
        ["analytical_quantities", "fit_summary", "intermediate_params"],
    )

    animation.create_animation(output, show_activity=False)
