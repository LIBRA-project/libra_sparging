from datetime import datetime
from pathlib import Path

from sparging.inputs import (
    LIBRA_PI_GEOM,
    LIBRA_PI_MAT,
    LIBRA_PI_OPERATING_PARAMS,
    LIBRA_PI_SPARGING_PARAMS,
    SimulationInput,
)
from sparging.model import Simulation, SimulationResults
from sparging.config import ureg
import logging
from dataclasses import replace

import matplotlib.pyplot as plt
import sparging.postprocess as pp
import numpy as np

from autoemulate.simulations.base import Simulator
from autoemulate import AutoEmulate
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis

import torch
import pandas as pd
import json


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

FOLDER = Path("training") / datetime.now().strftime("%Y%m%d_%H%M%S")
FOLDER.mkdir(exist_ok=True, parents=True)
FOLDER_SAMPLES = FOLDER / "samples"
FOLDER_SAMPLES.mkdir(exist_ok=True, parents=True)
FOLDER_PP = FOLDER / "postprocessing"
FOLDER_PP.mkdir(exist_ok=True)

outputs = []
t_irr = 8 * ureg.hour
t_sparging = 7 * ureg.day

librapi = SimulationInput.from_parameters(
    LIBRA_PI_GEOM.copy(),
    LIBRA_PI_MAT.copy(),
    LIBRA_PI_OPERATING_PARAMS.copy(),
    LIBRA_PI_SPARGING_PARAMS.copy(),
)

# for i, (P, T) in enumerate(samples):
#     librapi = SimulationInput.from_parameters(
#         LIBRA_PI_GEOM.copy(),
#         LIBRA_PI_MAT.copy(),
#         replace(LIBRA_PI_OPERATING_PARAMS, temperature=T, P_top=P),
#         LIBRA_PI_SPARGING_PARAMS.copy(),
#     )

#     tau = librapi.get_tau()
#     librapi.signal_irr = lambda t: 1 if t <= t_irr else 0
#     librapi.signal_sparging = lambda t: 0 if t <= t_irr else 1

#     my_simulation = Simulation(
#         librapi,
#         t_final=t_irr + t_sparging,
#     )
#     print(f"Running simulation {i} with P={P}, T={T}, tau={tau.to('hour')}")
#     outputs.append(my_simulation.solve(dt=8 * ureg.hour))

# get sim results from JSON file
# for i in range(len(samples)):
#     outputs.append(SimulationResults.from_json(f"output_{i}.json"))

# fig, ax = plt.subplots()
# residuals = []

# for i in range(len(outputs)):
#     res = pp.get_residual_fraction(
#         outputs[i].inventories_T2_salt, outputs[i].times, t_irr, t_irr + 1 * ureg.week
#     )
#     residuals.append(res)
#     print(res)


class SpargingProblem(Simulator):
    def __init__(self, parameters_range, output_names):
        self.counter = 0
        super().__init__(parameters_range, output_names)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # construct simulation input
        sim_input = librapi.copy()
        sim_input.h_l = np.power(10, x[0, 0].item()) * ureg("m/s")
        sim_input.eps_g = np.power(10, x[0, 1].item()) * ureg.dimensionless
        sim_input.a = x[0, 2].item() * ureg("m**-1")
        sim_input.temperature = x[0, 3].item() * ureg.celsius
        sim_input.K_s = np.power(10, x[0, 4].item()) * ureg(
            "mol/m**3/Pa"
        )  # express in molT2 ?
        sim_input.u_g0 = x[0, 5].item() * ureg("m/s")
        sim_input.signal_irr = lambda t: 1 if t <= t_irr else 0
        sim_input.signal_sparging = lambda t: 0 if t <= t_irr else 1

        # breakpoint()

        full_model = Simulation(
            sim_input,
            t_final=t_irr + t_sparging,
        )
        sim_output = full_model.solve(dt=0.2 * ureg.hour)
        sim_output.to_json(
            FOLDER_SAMPLES / f"sample_{self.counter}.json"
        )  # for debugging
        self.counter += 1

        residual_fraction = pp.get_residual_fraction(
            sim_output.inventories_T2_salt,
            sim_output.times,
            t_irr,
            t_irr + 1 * ureg.week,
        )
        PP_numbers.append(sim_input.get_PP_number().to("dimensionless").magnitude)
        y = torch.tensor(
            [[np.log10(residual_fraction.magnitude)]],
            dtype=torch.float64,
        )
        return y


simulator = SpargingProblem(
    # parameters_range={
    #     "log(h_l)": tuple(np.log10((1e-6, 1e-4))),
    #     "log(eps_g)": tuple(np.log10((1e-4, 2e-1))),
    #     "a": (0.05, 0.5),
    #     "temperature": (450, 800),
    #     "log(K_s)": tuple(np.log10((1e-6, 1e-1))),
    #     "u_g0": (0.02, 0.4),
    # },
    parameters_range={
        "log(h_l)": tuple(np.log10((1e-3, 1e-2))),
        "log(eps_g)": tuple(np.log10((1e-4, 2e-1))),
        "a": (0.05, 0.3),
        "temperature": (700, 800),
        "log(K_s)": tuple(np.log10((1e-3, 1e-1))),
        "u_g0": (0.02, 0.1),
    },
    output_names=["log(residual_1week)"],
)

n_samples = 50

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
pd.DataFrame(PP_numbers, columns=["PP_number"]).to_csv(
    FOLDER / "PP_numbers.csv", index=False
)

# Run AutoEmulate with default settings
ae = AutoEmulate(X, Y, log_level="WARNING")
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
problem = {
    "num_vars": simulator.in_dim,
    "names": simulator.param_names,
    "bounds": simulator.param_bounds,
    "output_names": simulator.output_names,
}

with open(FOLDER / "problem.json", "w") as f:
    json.dump(problem, f, indent=4)

sa = SensitivityAnalysis(emulator.model, problem=problem)
sobol_df = sa.run("sobol")
sa.plot_sobol(sobol_df, index="ST", fname=FOLDER_PP / "sobol.png")
