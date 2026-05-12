from __future__ import annotations
from sparging.config import ureg
from typing import TYPE_CHECKING
import numpy as np
import warnings
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pint
    from sparging.model import SimulationResults
    import matplotlib.pyplot as plt


def idx_from_t(times: list[pint.Quantity], timestamp: pint.Quantity) -> pint.Quantity:
    idx = np.argmin(np.abs(times - timestamp))
    time = times[idx].to(timestamp.units)
    if not np.isclose(times[idx], timestamp):
        warnings.warn(
            f"Requested time {timestamp} does not exactly match discrete times. Using closest time {time} for index calculation."
        )
    return idx


def plot_profile(
    sim_output: SimulationResults,
    ax: plt.Axes,
    var_name: str,
    times: list[pint.Quantity],
    colors: list = None,
    **kwargs,
):
    idx_to_plot = [idx_from_t(sim_output.times, t) for t in times]
    y_to_plot = getattr(sim_output, var_name)
    for j, idx in enumerate(idx_to_plot):
        ax.plot(
            sim_output.x_ct,
            y_to_plot[idx],
            label=f"t = {sim_output.times[idx]:.2f}",
            c=colors[j] if colors is not None else None,
            **kwargs,
        )


def get_residual_fraction(
    T2_inventories: np.ndarray[pint.Quantity],
    times: np.ndarray[pint.Quantity],
    t_0: pint.Quantity,
    t_end: pint.Quantity,
) -> pint.Quantity:
    i_0 = idx_from_t(times, t_0)
    i_end = idx_from_t(times, t_end)
    return T2_inventories[i_end] / T2_inventories[i_0]


def fit_exp(
    vec: np.ndarray[pint.Quantity],
    times: np.ndarray[pint.Quantity],
    t_0: pint.Quantity,
    t_end: pint.Quantity,
    phase: str,
) -> tuple[tuple[pint.Quantity, pint.Quantity], tuple[pint.Quantity, pint.Quantity]]:
    """
    Input:
    - vec: array of inventory values
    - times: array of times
    - t_0: initial fit time
    - t_end: final fit time
    - phase = 'decay' or 'rampup'
    ---
    Returns: (tau, n0), (tau_std, n0_std)
    """

    def fitting_func(t, tau, n0):
        match phase:
            case "decay":
                return n0 * np.exp(-t / tau)
            case "rampup":
                return n0 * (1 - np.exp(-t / tau))
            case _:
                raise ValueError("Invalid phase. Must be 'decay' or 'rampup'.")

    idx_0 = idx_from_t(times, t_0)
    idx_end = idx_from_t(times, t_end)
    t_0 = times[idx_0]
    t_end = times[idx_end]
    logger.info(
        f"Fitting from t={t_0.to('hour')} to t={t_end.to('hour')} (indices {idx_0} to {idx_end})"
    )
    tau_guess = 10000 * ureg.s
    n0_guess = vec[idx_0] if phase == "decay" else vec[idx_end]

    # wrapped_fitting_func = ureg.wraps(vec.units, (None, vec.units, "s"))(fitting_func)

    popt, pcov = curve_fit(
        fitting_func,
        (times[idx_0 : idx_end + 1] - t_0).to("s").magnitude,
        vec[idx_0 : idx_end + 1].magnitude,
        p0=[tau_guess.to("s").magnitude, n0_guess.magnitude],
    )

    # check goodness of fit
    tau_std = np.sqrt(pcov[0, 0])
    tau_rel_error = tau_std / popt[0]
    if tau_rel_error > 0.005:
        warnings.warn(
            f"High relative error in fitted tau: {tau_rel_error:.4f}. Fit may be unreliable."
        )
    # reattach units
    return (popt[0] * ureg.s, popt[1] * vec.units), (
        pcov[0] * ureg.s,
        pcov[1] * vec.units,
    )


def get_tau_real(
    vec: np.ndarray[pint.Quantities],
    times: np.ndarray[pint.Quantities],
    t_0: pint.Quantity,
) -> pint.Quantity:
    """
    To be distinguished from the time constant predicted from the model input parameters (tau analytical)
    Get the time constant tau from an exponential decay curve by finding the time at which the value has decayed to 1/e of its initial value at t_0.
    No exponential fit is performed here
    """
    idx_0 = idx_from_t(times, t_0)
    # Fit only the decay part
    vec_decay = vec[idx_0:]
    n0 = vec_decay[0]
    idx_tau = np.argmin(np.abs(vec_decay - n0 / np.e))
    tau = times[idx_0 + idx_tau] - t_0
    return tau
