from sparging.model import Simulation, SimulationResults
from sparging.inputs import (
    LIBRA_PI_GEOM,
    LIBRA_PI_MAT,
    LIBRA_PI_OPERATING_PARAMS,
    LIBRA_PI_SPARGING_PARAMS,
    SimulationInput,
    ureg,
)
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
    # serialization
    path_json = Path(tmp_path).joinpath("results.json")
    path_pkl = Path(tmp_path).joinpath("results.pkl")
    res.to_json(path_json)
    res.to_pickle(path_pkl)

    # deserialization
    new_res_json = SimulationResults.from_json(path_json)
    new_res_pickle = SimulationResults.from_pickle(path_pkl)

    # TEST
    assert len(res.times) == len(new_res_json.times), (
        "JSON Times arrays have different lengths"
    )
    assert len(res.c_T2_solutions) == len(new_res_json.c_T2_solutions), (
        "JSON c_T2_solutions arrays have different lengths"
    )

    assert np.allclose(res.times, new_res_json.times), "JSON Times arrays are not close"
    assert np.allclose(res.c_T2_solutions, new_res_json.c_T2_solutions), (
        "JSON c_T2_solutions arrays are not close"
    )

    assert len(res.times) == len(new_res_pickle.times), (
        "Pickle Times arrays have different lengths"
    )
    assert len(res.c_T2_solutions) == len(new_res_pickle.c_T2_solutions), (
        "Pickle c_T2_solutions arrays have different lengths"
    )
    assert np.allclose(res.times, new_res_pickle.times), (
        "Pickle Times arrays are not close"
    )
    assert np.allclose(res.c_T2_solutions, new_res_pickle.c_T2_solutions), (
        "Pickle c_T2_solutions arrays are not close"
    )
