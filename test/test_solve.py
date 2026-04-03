import sparging.model as model
from sparging.helpers import get_input
import pytest


def test_model_solve():
    """
    Tests that `model.solve` runs without errors for a simple test case. Does not check results.
    """
    params = get_input("test/test_input.yml")
    sim_input = model.SimulationInput(params)

    # TODO integrate to input file
    t_sparging_hr = [24, 1e20]  # time interval when sparger is ON
    t_irr_hr = [0, 96]  # time interval when irradiation is ON
    t_final = 1 * model.days_to_seconds

    model.solve(
        sim_input,
        t_final=t_final,
        t_irr=[t * model.hours_to_seconds for t in t_irr_hr],
        t_sparging=[t * model.hours_to_seconds for t in t_sparging_hr],
    )


def test_model_solve_incomplete_input():
    """
    Tests SimulationInput raises error when required input is missing.
    """
    params = get_input("test/test_input.yml")
    params.pop("D")  # remove bubble velocity to test default value

    with pytest.raises(KeyError, match="Missing a required input"):
        model.SimulationInput(params)
