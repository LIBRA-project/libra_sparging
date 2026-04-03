from __future__ import annotations
from sparging.config import ureg, const_R, const_g, VERBOSE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparging.model import SimulationInput
import numpy as np
import scipy.constants as const
import warnings

U_G0_DEFAULT = 0.25  # m/s, typical bubble velocity according to Chavez 2021


correlations_dict = {
    "rho_l": lambda input: ureg.Quantity(
        2245 - 0.424 * input.T.to("celsius").magnitude, "kg/m**3"
    ),  # density of Li2BeF4, Vidrio 2022
    "mu_l": lambda input: ureg.Quantity(
        0.116e-3 * np.exp(3755 / input.T.to("kelvin").magnitude), "Pa*s"
    ),  # kinematic viscosity of Li2BeF4, Cantor 1968
    "sigma_l": lambda input: ureg.Quantity(
        260 - 0.12 * input.T.to("celsius").magnitude, "dyn/cm"
    ).to("N/m"),  # surface tension of Li2BeF4,Cantor 1968
    "D_l": lambda input: ureg.Quantity(
        9.3e-7 * np.exp(-42e3 / (const.R * input.T.to("kelvin").magnitude)), "m**2/s"
    ),  # diffusivity of T in FLiBe, Calderoni 2008
    "K_s": lambda input: ureg.Quantity(
        7.9e-2 * np.exp(-35e3 / (const.R * input.T.to("kelvin").magnitude)),
        "mol/m**3/Pa",
    ),  # solubility of T in FLiBe, Calderoni 2008
    "d_b": lambda input: get_d_b(
        flow_g_vol=input.flow_g_vol,
        nozzle_diameter=input.nozzle_diameter,
        nb_nozzle=input.nb_nozzle,
    ),  # mean bubble diameter, Kanai 2017
    "Eo": lambda input: (const_g * input.drho * input.d_b**2 / input.sigma_l).to(
        "dimensionless"
    ),  # Eotvos number
    "Mo": lambda input: (
        input.drho * const_g * input.mu_l**4 / (input.rho_l**2 * input.sigma_l**3)
    ).to("dimensionless"),  # Morton number
    "Sc": lambda input: (input.nu_l / input.D_l).to("dimensionless"),  # Schmidt number
    "Re": lambda input: get_Re(input),  # Reynolds number
    "u_g0": lambda input: get_u_g0(input),  # initial gas velocity
    "eps_g": lambda input: get_eps_g(input),  # gas void fraction
    "h_l_malara": lambda input: get_h_malara(
        input.D_l, input.d_b
    ),  # mass transfer coefficient with Malara correlation
    "h_l_briggs": lambda input: get_h_briggs(
        input.Re, input.Sc, input.D_l, input.d_b
    ),  # mass transfer coefficient with Briggs correlation
    "h_l_higbie": lambda input: get_h_higbie(
        input.D_l, input.u_g0, input.d_b
    ),  # mass transfer coefficient with Higbie correlation
    "E_g": lambda input: get_E_g(
        input.tank_diameter, input.u_g0
    ),  # gas phase axial dispersion coefficient
    "source_T": lambda input: (
        input.tbr * input.n_gen_rate / (input.tank_volume)
    ),  # tritium generation source term
}


def get_d_b(flow_g_vol: float, nozzle_diameter: float, nb_nozzle: int) -> float:
    """
    mean bubble diameter [m], Kanai 2017 (reported by Evans 2026)
    """
    nozzle_flow = flow_g_vol / nb_nozzle  # volumetric flow per nozzle [m3/s]
    if nozzle_flow < ureg("3 cm**3/s") or nozzle_flow > ureg("10 cm**3/s"):
        warnings.warn(
            f"nozzle flow {nozzle_flow.to('cm**3/s')} is out of the validated range for the Kanai 2017 correlation (3-10 cm3/s)"
        )
    return ureg.Quantity(
        0.54
        * (
            nozzle_flow.to("cm**3/s").magnitude
            * np.sqrt(nozzle_diameter.to("cm").magnitude / 2)
        )
        ** 0.289,
        "cm",
    )


def get_Re(input: "SimulationInput") -> float:
    try:
        u = input.u_g0
        Re_old = input.Re
    except AttributeError:
        if VERBOSE:
            print("in get_Re, use default bubble velocity")
        u = U_G0_DEFAULT * ureg("m/s")
        Re_old = 0

    Re = input.rho_l * u * input.d_b / input.mu_l
    if (
        Re_old != 0 and np.abs(np.log10(Re / Re_old)) > 1
    ):  # check if Reynolds number changed by more than an order of magnitude
        warnings.warn(
            f"Re number changed significantly from {Re_old:.2e} to {Re:.2e}, bubble velocity u = {u} might be off. Check assumed bubble velocity in initial Reynolds number calculation"
        )
    return Re.to("dimensionless")


def get_u_g0(input: "SimulationInput") -> float:  # TODO move inside class ?
    """
    bubble initial velocity [m/s], correlation for terminal velocity from Clift 1978
    """
    H = (4 / 3 * input.Eo.magnitude * input.Mo.magnitude**-0.149) * (
        input.mu_l.magnitude / 0.0009
    ) ** -0.14
    if H > 59.3:
        J = 3.42 * H**0.441
    elif H > 2:
        J = 0.94 * H**0.757
    else:
        J = input.Re * input.Mo**0.149 + 0.857
        warnings.warn(
            f"Warning: low Reynolds number {input.Re:.2e}, bubble size d_b = {input.d_b} m might be too small."
            f"Clift correlation will use default value for bubble velocity u_g0 = {U_G0_DEFAULT} m/s"
        )
    u_g0 = input.mu_l / (input.rho_l * input.d_b) * input.Mo**-0.149 * (J - 0.857)
    if u_g0 > ureg("1 m/s") or u_g0 < ureg("0.1 m/s"):
        warnings.warn(f"Warning: bubble velocity {u_g0} is out of the typical range")

    return u_g0


def get_eps_g(input: "SimulationInput") -> float:
    """computes gas void fraction from ideal gas law and Young-Laplace pressure in the bubbles (neglecting hydrostatic pressure variation)"""
    eps_g = (
        const_R
        * input.T
        / (input.P_0 + 4 * input.sigma_l / input.d_b)
        * input.flow_g
        / (np.pi * (input.tank_diameter / 2) ** 2 * input.u_g0)
    )
    if eps_g > 1 or eps_g < 0:
        warnings.warn(f"Warning: unphysical gas fraction: {eps_g}")
    elif eps_g > 0.1:
        warnings.warn(
            f"Warning: high gas fraction: {eps_g}, models assumptions may not hold"
        )
    return eps_g


def get_h_higbie(D_l: float, u_g: float, d_b: float) -> float:
    """mass transfer coefficient [m/s] for tritium in liquid FLiBe using Higbie penetration model"""
    h_l = (
        (D_l * u_g) / (const.pi * d_b)
    ) ** 0.5  # mass transport coefficient Higbie penetration model
    return h_l


def get_h_malara(D_l: float, d_b: float) -> float:
    """
    mass transfer coefficient [m/s] for tritium in liquid FLiBe using Malara 1995 correlation
    (used for inert gas stripping from breeder droplets, may not be valid here)
    """
    h_l = 2 * np.pi**2 * D_l / (3 * d_b)
    return h_l


def get_h_briggs(Re: float, Sc: float, D_l: float, d_b: float) -> float:
    """mass transfer coefficient [m/s] for tritium in liquid FLiBe using Briggs 1970 correlation"""
    Sh = 0.089 * Re**0.69 * Sc**0.33  # Sherwood number
    h_l = Sh * D_l / d_b
    return h_l


def get_E_g(diameter: float, u_g: float) -> float:
    """gas phase axial dispersion coefficient [m2/s], Malara 1995 correlation
    models dispersion of the gas velocity distribution around the mean bubble velocity"""
    E_g = 0.2 * ureg("1/m") * diameter**2 * u_g
    return E_g
