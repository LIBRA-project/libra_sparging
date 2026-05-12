from datetime import datetime
from pathlib import Path

from sparging import (
    LIBRA_PI_GEOM,
    LIBRA_PI_MAT,
    LIBRA_PI_OPERATING_PARAMS,
    LIBRA_PI_SPARGING_PARAMS,
    SimulationInput,
    ureg,
    all_correlations,
    CorrelationType,
)

import logging
import pandas as pd
import json
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

FOLDER = Path("datasets") / datetime.now().strftime("%Y%m%d_%H%M%S")
FOLDER.mkdir(exist_ok=True, parents=True)
FOLDER_SAMPLES = FOLDER / "samples"
FOLDER_SAMPLES.mkdir(exist_ok=True, parents=True)
FOLDER_PP = FOLDER / "postprocessing"
FOLDER_PP.mkdir(exist_ok=True)

outputs = []
h_l_corrs = all_correlations.get_list(CorrelationType.MASS_TRANSFER_COEFF)
D_l_corrs = all_correlations.get_list(CorrelationType.DIFFUSIVITY)


def forward(count, x: dict) -> list:
    """return list of values for post processing"""
    # construct simulation input
    LIBRA_PI_OPERATING_PARAMS.temperature = x["temperature"] * ureg.celsius
    LIBRA_PI_OPERATING_PARAMS.P_top = x["P_top"] * ureg.bar
    LIBRA_PI_OPERATING_PARAMS.flow_g_mol = x["flow_g_mol"] * ureg.sccm
    LIBRA_PI_GEOM.nozzle_diameter = x["nozzle_diameter"] * ureg.m
    # LIBRA_PI_SPARGING_PARAMS.h_l = x["h_l_corr"]
    LIBRA_PI_MAT.D_l = D_l_corrs[x["D_l_corr_number"]]
    # breakpoint()

    graph = nx.Graph()
    sim_input = SimulationInput.from_parameters(
        LIBRA_PI_GEOM,
        LIBRA_PI_MAT,
        LIBRA_PI_OPERATING_PARAMS,
        LIBRA_PI_SPARGING_PARAMS,
        graph=graph,
    )
    tau = sim_input.get_tau()
    h_l = sim_input.h_l
    a = sim_input.a
    eps_g = sim_input.eps_g
    # sim_input.to_json(
    #     FOLDER_SAMPLES / f"sample_{count}.json"
    # )  # for debugging and postprocessing

    # for post processing
    Pi = sim_input.get_Pi_number().to("dimensionless").magnitude
    return [
        Pi,
        tau.to("s").magnitude,
        h_l.to("m/s").magnitude,
        a.to("1/m").magnitude,
        eps_g.to("dimensionless").magnitude,
        graph.nodes["d_b"]["value"].to("m").magnitude,
        graph.nodes["u_g0"]["value"].to("m/s").magnitude,
        graph.nodes["h_l"]["origin"],
        graph.nodes["D_l"]["origin"],
    ]


n_samples = 1000
inputs = []
postprocess = []

for i in range(n_samples):
    input = {
        "temperature": np.random.normal(550, 25),  # celsius
        "P_top": 1.2,  # bar
        "flow_g_mol": 400,  # sccm
        "nozzle_diameter": 1.5e-3,  # m
        "D_l_corr_number": np.random.randint(
            0, 2
        ),  # index for selecting diffusivity correlation
    }
    inputs.append(input)
    print(f"Running simulation {i + 1}/{n_samples}")
    postprocess.append(forward(i, input))

# save training data
PP_frame = pd.DataFrame(
    postprocess,
    columns=["Pi", "tau", "h_l", "a", "eps_g", "d_b", "u_g0", "h_l_corr", "D_l_corr"],
)
PP_frame.to_csv(FOLDER / "PP_data.csv", index=False)

# for compatibility with postprocessing code
pd.DataFrame(PP_frame, columns=["tau"]).to_csv(
    FOLDER / "simulator_outputs.csv", index=False
)
input_frame = pd.DataFrame(inputs)
input_frame.to_csv(FOLDER / "simulator_inputs.csv", index=False)

problem = {
    "names": [x for x in inputs[0].keys()],
    "bounds": ["550 +- 25", "1.2", "400", "1.5e-3", "oishi or calderoni"],
}

with open(FOLDER / "problem.json", "w") as f:
    json.dump(problem, f, indent=4)
