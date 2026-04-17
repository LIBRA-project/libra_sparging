from __future__ import annotations
from sparging.config import ureg
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import pint
    from sparging.model import SimulationResults
    import matplotlib.pyplot as plt


def indices_from_times(sim_output: SimulationResults, times_ref: list[pint.Quantity]):
    i_matching = [np.argmin(np.abs(sim_output.times - t)) for t in times_ref]
    times_matching = [sim_output.times[i] for i in i_matching]
    return i_matching, times_matching


def plot_profile(
    sim_output: SimulationResults,
    ax: plt.Axes,
    var_name: str,
    times: list[pint.Quantity],
):
    i_to_plot, _ = indices_from_times(sim_output, times)
    y_to_plot = getattr(sim_output, var_name)
    for i in i_to_plot:
        ax.plot(
            sim_output.x_ct,
            y_to_plot[i],
            label=f"t = {sim_output.times[i] / 3600:.2f} h",
        )
    ax.legend()
    ax.set_xlabel(r"z [m]")
    ax.set_ylabel(var_name)  # TODO leverage pint for units
    ax.set_title(var_name + " profile")
    ax.grid()


def plot_signal(sim_output: SimulationResults, ax: plt.Axes, var_name: str):
    y_to_plot = getattr(sim_output, var_name)
    ax.plot(
        sim_output.times,
        y_to_plot,
        label=var_name,
    )
    ax.legend()
    ax.set_xlabel(r"t [s]")
    ax.set_ylabel(var_name)  # TODO leverage pint for units
    ax.set_title(var_name + " profile")
    ax.grid()
