import sparging.model as model
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


def test_model_solve_successfull(tmp_path):
    """
    Tests that `model.solve` runs without errors for a simple test case. Does not check results.
    Also tests successful exporting results to yaml, json and csv files.
    """
    # TODO integrate to input file
    t_sparging = [2, 1e20] * ureg.hours  # time interval when sparger is ON
    t_irr = [1, 3] * ureg.hours  # time interval when irradiation is ON
    t_final = 0.2 * ureg.day

    output = model.solve(
        standard_input,
        t_final=t_final,
        t_irr=t_irr,
        t_sparging=t_sparging,
    )
    from pathlib import Path

    output.to_yaml(Path(tmp_path).joinpath("dummy.yaml"))
    output.to_json(Path(tmp_path).joinpath("dummy.json"))
    output.profiles_to_csv(Path(tmp_path))


def test_model_solve_missing_input():
    """
    Tests SimulationInput raises error when required input is missing.
    """
    broken_input = dataclasses.replace(standard_input)
    del broken_input.u_g0  # missing required parameter

    t_sparging = [0, 1e20] * ureg.hours  # time interval when sparger is ON
    t_irr = [0, 4] * ureg.hours  # time interval when irradiation is ON
    t_final = 1 * ureg.day

    with pytest.raises(
        AttributeError, match="'SimulationInput' object has no attribute 'u_g0'"
    ):
        model.solve(
            broken_input,
            t_final=t_final,
            t_irr=t_irr,
            t_sparging=t_sparging,
        )


def test_model_solve_wrong_input():
    """
    Tests SimulationInput raises error when required input has wrong dimensionality
    """
    broken_input = dataclasses.replace(
        standard_input, u_g0=3 * ureg("m^2/s")
    )  # wrong units

    t_sparging = [0, 1e20] * ureg.hours  # time interval when sparger is ON
    t_irr = [0, 36] * ureg.hours  # time interval when irradiation is ON
    t_final = 1 * ureg.day

    with pytest.raises(DimensionalityError, match="Cannot convert from"):
        model.solve(
            broken_input,
            t_final=t_final,
            t_irr=t_irr,
            t_sparging=t_sparging,
        )
