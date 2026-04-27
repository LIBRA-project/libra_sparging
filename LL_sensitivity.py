from datetime import datetime
from pathlib import Path

from sparging.inputs import (
    LIBRA_PI_GEOM,
    LIBRA_PI_MAT,
    LIBRA_PI_OPERATING_PARAMS,
    LIBRA_PI_SPARGING_PARAMS,
    SimulationInput,
)
from sparging.config import ureg
from sparging.correlations import all_correlations, CorrelationType

import logging

from autoemulate.simulations.base import Simulator
from autoemulate import AutoEmulate
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis

import torch
import pandas as pd
import json

COMPUTE_SOBOL = False

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


class SpargingProblem(Simulator):
    def __init__(self, parameters_range, output_names):
        self.counter = 0
        super().__init__(parameters_range, output_names)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # construct simulation input
        LIBRA_PI_OPERATING_PARAMS.temperature = x[0, 0].item() * ureg.celsius
        LIBRA_PI_OPERATING_PARAMS.P_top = x[0, 1].item() * ureg.bar
        LIBRA_PI_OPERATING_PARAMS.flow_g_mol = x[0, 2].item() * ureg.sccm
        LIBRA_PI_GEOM.nozzle_diameter = x[0, 3].item() * ureg.m
        LIBRA_PI_SPARGING_PARAMS.h_l = h_l_corrs[int(x[0, 4].item())]
        # breakpoint()

        sim_input = SimulationInput.from_parameters(
            LIBRA_PI_GEOM,
            LIBRA_PI_MAT,
            LIBRA_PI_OPERATING_PARAMS,
            LIBRA_PI_SPARGING_PARAMS,
        )
        tau = sim_input.get_tau()
        h_l = sim_input.h_l
        a = sim_input.a
        eps_g = sim_input.eps_g
        # sim_input.to_json(
        #     FOLDER_SAMPLES / f"sample_{self.counter}.json"
        # )  # for debugging and postprocessing
        self.counter += 1

        # for post processing
        Pi = sim_input.get_Pi_number().to("dimensionless").magnitude
        PP_numbers.append(
            [
                Pi,
                tau.to("s").magnitude,
                h_l.to("m/s").magnitude,
                a.to("1/m").magnitude,
                eps_g.to("dimensionless").magnitude,
                LIBRA_PI_SPARGING_PARAMS.h_l.identifier,
            ]
        )
        y = torch.tensor(
            [[tau.to("s").magnitude]],
            dtype=torch.float64,
        )
        return y


simulator = SpargingProblem(
    parameters_range={  # realistic (wide) parameters range for LIBRA
        "temperature": (450, 800),  # celsius
        "P_top": (1, 5),  # bar
        "flow_g_mol": (40, 400),  # sccm
        "nozzle_diameter": (0.5e-3, 10e-3),  # m
        "h_l_corr": (0, len(h_l_corrs)),
    },
    output_names=["tau"],
)

n_samples = 5000

X = simulator.sample_inputs(n_samples)

PP_numbers = []
Y, _ = simulator.forward_batch(X, allow_failures=False)

# save training data
pd.DataFrame(Y, columns=simulator.output_names).to_csv(
    FOLDER / "simulator_outputs.csv", index=False
)
pd.DataFrame(X, columns=simulator.param_names).to_csv(
    FOLDER / "simulator_inputs.csv", index=False
)
pd.DataFrame(PP_numbers, columns=["Pi", "tau", "h_l", "a", "eps_g", "h_l_corr"]).to_csv(
    FOLDER / "PP_data.csv", index=False
)

# sensitivity analysis problem
problem = {
    "num_vars": simulator.in_dim,
    "names": simulator.param_names,
    "bounds": simulator.param_bounds,
    "output_names": simulator.output_names,
}

with open(FOLDER / "problem.json", "w") as f:
    json.dump(problem, f, indent=4)

if COMPUTE_SOBOL:
    # Run AutoEmulate with default settings
    ae = AutoEmulate(X, Y, log_level="WARNING", models=["GaussianProcessRBF"])
    ae.summarise()

    # pick best model
    emulator = ae.best_result()
    print(f"Selected model: {emulator.model_name} with id: {emulator.id}")

    # The use_timestamp paramater ensures a new result is saved each time the save method is called
    best_result_filepath = ae.save(emulator, FOLDER, use_timestamp=False)
    print("Model and metadata saved to: ", best_result_filepath)

    ae.plot_preds(
        emulator,
        output_names=simulator.output_names,
        fname=FOLDER_PP / "predictions.png",
    )

    # === Sensitivity analysis ===
    sa = SensitivityAnalysis(emulator.model, problem=problem)
    sobol_df = sa.run("sobol")
    sa.plot_sobol(sobol_df, index="ST", fname=FOLDER_PP / "sobol.png")
