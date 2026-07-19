from sparging import (
    LIBRA_PI_GEOM,
    LIBRA_PI_MAT,
    LIBRA_PI_OPERATING_PARAMS,
    LIBRA_PI_SPARGING_PARAMS,
    SimulationInput,
    ureg,
    Simulation,
    SimulationResults,
)
import json
import numpy as np
from pathlib import Path


def test_simulation_results_serialization(tmp_path):
    # BUILD
    my_input = SimulationInput.from_parameters(
        LIBRA_PI_GEOM,
        LIBRA_PI_MAT,
        LIBRA_PI_OPERATING_PARAMS,
        LIBRA_PI_SPARGING_PARAMS,
    )

    my_sim = Simulation(
        my_input,
        t_final=2 * ureg.hour,
    )

    res = my_sim.solve(fast_solve=True)

    # RUN
    # Full serialization round-trips through pickle; to_json now writes a scalar
    # SUMMARY (selected sections), not the full profiles -- see to_json docstring.
    path_json = Path(tmp_path).joinpath("summary.json")
    path_pkl = Path(tmp_path).joinpath("results.pkl")
    res.to_pickle(path_pkl)
    res.to_json(path_json, ["analytical_quantities", "intermediate_params"])

    new_res_pickle = SimulationResults.from_pickle(path_pkl)

    # TEST
    # pickle: full round-trip of the arrays
    assert len(res.times) == len(new_res_pickle.times), (
        "Pickle Times arrays have different lengths"
    )
    assert len(res.c_T2_profiles) == len(new_res_pickle.c_T2_profiles), (
        "Pickle c_T2_profiles arrays have different lengths"
    )
    assert np.allclose(res.times, new_res_pickle.times, atol=0), (
        "Pickle Times arrays are not close"
    )
    assert np.allclose(res.c_T2_profiles, new_res_pickle.c_T2_profiles, atol=0), (
        "Pickle c_T2_profiles arrays are not close"
    )

    # json: a valid scalar summary containing the requested sections
    with open(path_json) as f:
        summary = json.load(f)
    assert "analytical_quantities" in summary and "intermediate_params" in summary, (
        "JSON summary is missing the requested sections"
    )
