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
    tau_guess: pint.Quantity = 10000 * ureg.s,
) -> tuple[tuple[pint.Quantity, pint.Quantity], tuple[pint.Quantity, pint.Quantity]]:
    """
    Input:
    - vec: array of inventory values
    - times: array of times
    - t_0: initial fit time
    - t_end: final fit time
    - phase = 'decay' or 'rampup'
    - tau_guess: initial guess for tau passed to curve_fit. The default (10000 s)
      is a historical arbitrary value; pass the analytical tau_pred when the
      true timescale may be far from that (curve_fit can otherwise fail to
      converge, especially when the fit window only spans a couple of tau).
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
    n0_std = np.sqrt(pcov[1, 1])
    tau_rel_error = tau_std / popt[0]
    if tau_rel_error > 0.005:
        warnings.warn(
            f"High relative error in fitted tau: {tau_rel_error:.4f}. Fit may be unreliable."
        )
    return (popt[0] * ureg.s, popt[1] * vec.units), (
        tau_std * ureg.s,
        n0_std * vec.units,
    )


def get_exp_fit_rmse(
    vec: np.ndarray[pint.Quantities],
    times: np.ndarray[pint.Quantities],
    t_0: pint.Quantity,
    t_end: pint.Quantity,
    tau: pint.Quantity,
    n0: pint.Quantity,
) -> pint.Quantity:
    """Normalized RMSE (RMSE / n0) of an exponential-decay fit (tau, n0) against
    vec over [t_0, t_end]. Fit-quality diagnostic: large values indicate the
    decay is not well described by a single exponential."""
    idx_0 = idx_from_t(times, t_0)
    idx_end = idx_from_t(times, t_end)
    t = (times[idx_0 : idx_end + 1] - times[idx_0]).to("s")
    fitted = n0 * np.exp(-t / tau.to("s"))
    actual = vec[idx_0 : idx_end + 1]
    rmse = np.sqrt(np.mean((actual - fitted) ** 2))
    return (rmse / n0).to("dimensionless")


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


def summarize_decay(
    results: SimulationResults,
    t_0: pint.Quantity | None = None,
    t_end: pint.Quantity | None = None,
) -> dict:
    """Fit n_T2_salt(t) with a decaying exponential over [t_0, t_end] and
    compare with the analytical prediction. Returns a dict of pint.Quantities."""
    times = results.times
    inv = results.n_T2_salt_series

    if t_0 is None:  # default: decay starts at the inventory peak
        t_0 = times[int(np.argmax(inv))]
    if t_end is None:
        t_end = times[-1]

    (tau_fit, n0_fit), (tau_std, n0_std) = fit_exp(inv, times, t_0, t_end, "decay")
    tau_real = get_tau_real(inv, times, t_0)
    tau_pred = results.sim_input.get_tau()
    tau_pred_ave = results.sim_input.get_tau_ave()
    residual = get_residual_fraction(inv, times, t_0, t_end)
    rel_error = ((tau_fit - tau_pred) / tau_pred).to("dimensionless")
    ave_rel_error = ((tau_fit - tau_pred_ave) / tau_pred_ave).to("dimensionless")

    logger.info(
        f"tau_fitted={tau_fit.to('hour'):.3f}, tau_predicted={tau_pred.to('hour'):.3f}, "
        f"tau_real(1/e)={tau_real.to('hour'):.3f}, rel_error={rel_error.magnitude:.2%}"
    )

    return {
        "tau_fitted": tau_fit,
        "tau_fitted_std": tau_std,
        "tau_predicted": tau_pred,
        "tau_predicted_ave": tau_pred_ave,
        "tau_real_1e": tau_real,
        "n0_fitted": n0_fit,
        "n0_fitted_std": n0_std,
        "residual_fraction": residual,
        "tau_rel_error": rel_error,
        "tau_ave_rel_error": ave_rel_error,
        "fit_window": (t_0, t_end),
    }
