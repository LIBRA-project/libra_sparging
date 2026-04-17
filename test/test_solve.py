from sparging.model import Simulation
from sparging.inputs import SimulationInput
from sparging.config import ureg
import pytest
import dataclasses
from pint import DimensionalityError


standard_input = SimulationInput(
    height=1.0 * ureg.m,
    area=0.2 * ureg.m**2,
    u_g0=0.25 * ureg("m/s"),
    temperature=600 * ureg.celsius,
    a=0.5 * ureg("1/m"),
    h_l=3e-5 * ureg("m/s"),
    K_s=1e-4 * ureg("mol/m**3/Pa"),
    P_bottom=1.2 * ureg.bar,
    eps_g=0.001 * ureg.dimensionless,
    E_g=1e-2 * ureg("m^2/s"),
    D_l=3e-9 * ureg("m^2/s"),
    source_T=8e-16 * ureg("molT/m^3/s"),
)


@pytest.fixture
def standard_simulation():
    return Simulation(
        standard_input,
        t_final=6 * ureg.hours,
        signal_irr=lambda t: 1 if t > 1 * ureg.hour and t < 3 * ureg.hour else 0,
        signal_sparging=lambda t: 1,
    )


def test_model_solve_successfull(tmp_path, standard_simulation):
    """
    Tests that `model.solve` runs without errors for a simple test case. Does not check results.
    Also tests successful exporting results to yaml, json and csv files.
    """

    output = standard_simulation.solve(dt=0.05 * ureg.hour, dx=0.01 * ureg.m)
    from pathlib import Path

    output.to_yaml(Path(tmp_path).joinpath("dummy.yaml"))
    output.to_json(Path(tmp_path).joinpath("dummy.json"))
    output.profiles_to_csv(Path(tmp_path))


def test_model_solve_missing_input(standard_simulation):
    """
    Tests SimulationInput raises error when a required input quantity is missing.
    """
    # BUILD
    broken_input = dataclasses.replace(standard_input)
    del broken_input.u_g0  # missing required parameter
    standard_simulation.sim_input = broken_input

    # TEST
    with pytest.raises(
        AttributeError, match="'SimulationInput' object has no attribute 'u_g0'"
    ):
        standard_simulation.solve(dt=0.05 * ureg.hour, dx=0.01 * ureg.m)


def test_model_solve_wrong_input(standard_simulation):
    """
    Tests Simulation.solve() raises error when required input has wrong dimensionality
    """
    # BUILD
    broken_input = dataclasses.replace(
        standard_input, u_g0=3 * ureg("m^2/s")
    )  # wrong units
    standard_simulation.sim_input = broken_input

    # TEST
    with pytest.raises(DimensionalityError, match="Cannot convert from"):
        standard_simulation.solve(dt=0.05 * ureg.hour, dx=0.01 * ureg.m)


def test_model_solve_wrong_argument(standard_simulation):
    """
    Tests Simulation.solve() can't be given a timestep without specifying the units
    """
    with pytest.raises(AttributeError, match="object has no attribute 'to'"):
        standard_simulation.solve(dt=0.01)
