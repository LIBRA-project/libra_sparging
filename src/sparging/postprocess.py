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

    assert len(i_matching) == len(times_ref)
    return i_matching, times_matching


def plot_profile(
    sim_output: SimulationResults,
    ax: plt.Axes,
    var_name: str,
    times: list[pint.Quantity],
    colors: list = None,
    **kwargs,
):
    i_to_plot, _ = indices_from_times(sim_output, times)
    y_to_plot = getattr(sim_output, var_name)
    for j, idx in enumerate(i_to_plot):
        ax.plot(
            sim_output.x_ct,
            y_to_plot[idx],
            label=f"t = {sim_output.times[idx]:.2f}",
            c=colors[j] if colors is not None else None,
            **kwargs,
        )
    ax.legend()
    ax.set_xlabel(r"z [m]")
    ax.set_ylabel(var_name)  # TODO leverage pint for units
    ax.set_title(var_name + " profile")
    ax.grid()


def plot_signal(sim_output: SimulationResults, ax: plt.Axes, var_name: str, **kwargs):
    ax.plot(sim_output.times, getattr(sim_output, var_name), **kwargs)
    # ax.set_xlabel(r"t [s]")
    # ax.set_ylabel(var_name)  # TODO leverage pint for units
    # ax.set_title(var_name + " profile")
    # ax.grid()
