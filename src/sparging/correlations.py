from __future__ import annotations
from sparging.config import ureg, const_R, const_g, VERBOSE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparging.model import SimulationInput
import numpy as np
import scipy.constants as const
import warnings

from dataclasses import dataclass
import enum


class CorrelationType(enum.Enum):
    MASS_TRANSFER_COEFF = "h_l"
    DENSITY = "rho_l"
    DIFFUSIVITY = "D_l"
    SOLUBILITY = "K_S"
    VISCOSITY = "mu"
    SURFACE_TENSION = "sigma"
    GAS_VOID_FRACTION = "eps_g"
    BUBBLE_DIAMETER = "d_b"
    EOTVOS_NUMBER = "Eo"
    MORTON_NUMBER = "Mo"
    SCHMIDT_NUMBER = "Sc"
    REYNOLDS_NUMBER = "Re"
    BUBBLE_VELOCITY = "u_g0"
    GAS_PHASE_DISPERSION = "E_g"


@dataclass
class Correlation:
    identifier: str
    function: callable
    corr_type: CorrelationType
    source: str | None = None
    description: str | None = None
    input_units: list[str] | None = None

    def __call__(self, **kwargs):

        # check the dimensions are correct
        if self.input_units is not None:
            for arg_name, expected_dimension in zip(kwargs, self.input_units):
                arg = kwargs[arg_name]
                if not isinstance(arg, ureg.Quantity):
                    raise ValueError(
                        f"Invalid input: expected a pint.Quantity with units of {expected_dimension}, got {arg} of type {type(arg)}"
                    )
                if not arg.check(expected_dimension):
                    raise ValueError(
                        f"Invalid input: expected dimensions of {expected_dimension}, got {arg.dimensionality}"
                    )
        return self.function(**kwargs).to_base_units()

    # TODO add a method that checks the validity of the input parameters based on the range of validity of the correlation, if provided in the description or source. This method could be called before running the simulation to warn the user if they are using a correlation outside of its validated range.


class CorrelationGroup(list[Correlation]):
    def __call__(self, identifier: str) -> Correlation:
        for corr in self:
            if corr.identifier == identifier:
                return corr
        raise ValueError(f"Correlation with identifier {identifier} not found in group")


correlations = CorrelationGroup([])

U_G0_DEFAULT = 0.25  # m/s, typical bubble velocity according to Chavez 2021

rho_l = Correlation(
    identifier="rho_l",
    function=lambda temperature: ureg.Quantity(
        2245 - 0.424 * temperature.to("celsius").magnitude, "kg/m**3"
    ),  # density of Li2BeF4, Vidrio 2022
    corr_type=CorrelationType.DENSITY,
    source="Vidrio 2022",
    description="density of Li2BeF4 as a function of temperature",
    input_units=["kelvin"],
)
correlations.append(rho_l)

mu_l = Correlation(
    identifier="mu_l",
    function=lambda temperature: ureg.Quantity(
        0.116e-3 * np.exp(3755 / temperature.to("kelvin").magnitude), "Pa*s"
    ),  # kinematic viscosity of Li2BeF4, Cantor 1968
    corr_type=CorrelationType.VISCOSITY,
    source="Cantor 1968",
    description="dynamic viscosity of Li2BeF4 as a function of temperature",
    input_units=["kelvin"],
)
correlations.append(mu_l)

sigma_l = Correlation(
    identifier="sigma_l",
    function=lambda temperature: ureg.Quantity(
        260 - 0.12 * temperature.to("celsius").magnitude, "dyn/cm"
    ).to("N/m"),  # surface tension of Li2BeF4,Cantor 1968
    corr_type=CorrelationType.SURFACE_TENSION,
    source="Cantor 1968",
    description="surface tension of Li2BeF4 as a function of temperature",
    input_units=["kelvin"],
)
correlations.append(sigma_l)

# TODO this could leverage HTM
D_l = Correlation(
    identifier="D_l",
    function=lambda temperature: ureg.Quantity(
        9.3e-7 * np.exp(-42e3 / (const_R * temperature.to("kelvin").magnitude)),
        "m**2/s",
    ),  # diffusivity of T in FLiBe, Calderoni 2008
    corr_type=CorrelationType.DIFFUSIVITY,
    source="Calderoni 2008",
    description="diffusivity of tritium in liquid FLiBe as a function of temperature",
    input_units=["kelvin"],
)
correlations.append(D_l)

K_s = Correlation(
    identifier="K_s",
    function=lambda temperature: ureg.Quantity(
        7.9e-2 * np.exp(-35e3 / (const_R * temperature.to("kelvin").magnitude)),
        "mol/m**3/Pa",
    ),  # solubility of T in FLiBe, Calderoni 2008
    corr_type=CorrelationType.SOLUBILITY,
    source="Calderoni 2008",
    description="solubility of tritium in liquid FLiBe as a function of temperature",
    input_units=["kelvin"],
)
correlations.append(K_s)

d_b = Correlation(
    identifier="d_b",
    function=lambda flow_g_vol, nozzle_diameter, nb_nozzle: get_d_b(
        flow_g_vol=flow_g_vol, nozzle_diameter=nozzle_diameter, nb_nozzle=nb_nozzle
    ),  # mean bubble diameter, Kanai 2017
    corr_type=CorrelationType.BUBBLE_DIAMETER,
    input_units=["m**3/s", "m", "dimensionless"],
)
correlations.append(d_b)

E_o = Correlation(
    identifier="Eo",
    function=lambda drho, d_b, sigma_l: (const_g * drho * d_b**2 / sigma_l).to(
        "dimensionless"
    ),  # Eotvos number
    corr_type=CorrelationType.EOTVOS_NUMBER,
    input_units=["kg/m**3", "m", "N/m"],
)
correlations.append(E_o)

Mo = Correlation(
    identifier="Mo",
    function=lambda drho, mu_l, rho_l, sigma_l: (
        drho * const_g * mu_l**4 / (rho_l**2 * sigma_l**3)
    ).to("dimensionless"),  # Morton number
    corr_type=CorrelationType.MORTON_NUMBER,
    input_units=["kg/m**3", "Pa*s", "kg/m**3", "N/m"],
)
correlations.append(Mo)

Sc = Correlation(
    identifier="Sc",
    function=lambda nu_l, D_l: (nu_l / D_l).to("dimensionless"),  # Schmidt number
    corr_type=CorrelationType.SCHMIDT_NUMBER,
    input_units=["m**2/s", "m**2/s"],
)
correlations.append(Sc)

# NOTE on temporise
# Re = Correlation(
#     identifier="Re",
#     function=lambda input: get_Re(input),  # Reynolds number
#     corr_type=CorrelationType.REYNOLDS_NUMBER,
#     input_units=None,  # TODO this correlation needs access to multiple input parameters, need to figure out how to handle this
# )

u_g0 = Correlation(
    identifier="u_g0",
    function=lambda Eo, Mo, Re, mu_l, rho_l, d_b: get_u_g0(
        Eo=Eo, Mo=Mo, Re=Re, mu_l=mu_l, rho_l=rho_l, d_b=d_b
    ),  # initial gas velocity
    corr_type=CorrelationType.BUBBLE_VELOCITY,
    input_units=[
        "dimensionless",
        "dimensionless",
        "dimensionless",
        "Pa*s",
        "kg/m**3",
        "m",
    ],
)
correlations.append(u_g0)

eps_g = Correlation(
    identifier="eps_g",
    function=lambda temperature, P_0, sigma_l, d_b, flow_g, tank_diameter, u_g0: (
        get_eps_g(
            T=temperature,
            P_0=P_0,
            sigma_l=sigma_l,
            d_b=d_b,
            flow_g=flow_g,
            tank_diameter=tank_diameter,
            u_g0=u_g0,
        )
    ),  # gas void fraction
    corr_type=CorrelationType.GAS_VOID_FRACTION,
    input_units=[
        "kelvin",
        "Pa",
        "N/m",
        "m",
        "m**3/s",
        "m",
        "m/s",
    ],
)
correlations.append(eps_g)

h_l_higbie = Correlation(
    identifier="h_l_higbie",
    function=lambda D_l, u_g, d_b: get_h_higbie(
        D_l=D_l, u_g=u_g, d_b=d_b
    ),  # mass transfer coefficient with Higbie correlation
    corr_type=CorrelationType.MASS_TRANSFER_COEFF,
    source="Higbie 1935",
    description="mass transfer coefficient for tritium in liquid FLiBe using Higbie penetration model",
    input_units=["m**2/s", "m/s", "m"],
)
correlations.append(h_l_higbie)

h_l_malara = Correlation(
    identifier="h_l_malara",
    function=lambda D_l, d_b: get_h_malara(
        D_l=D_l, d_b=d_b
    ),  # mass transfer coefficient with Malara correlation
    corr_type=CorrelationType.MASS_TRANSFER_COEFF,
    source="Malara 1995",
    description="mass transfer coefficient for tritium in liquid FLiBe using Malara 1995 correlation (used for inert gas stripping from breeder droplets, may not be valid here)",
    input_units=["m**2/s", "m"],
)
correlations.append(h_l_malara)

h_l_briggs = Correlation(
    identifier="h_l_briggs",
    function=lambda Re, Sc, D_l, d_b: get_h_briggs(
        Re=Re, Sc=Sc, D_l=D_l, d_b=d_b
    ),  # mass transfer coefficient with Briggs correlation
    corr_type=CorrelationType.MASS_TRANSFER_COEFF,
    source="Briggs 1970",
    description="mass transfer coefficient for tritium in liquid FLiBe using Briggs 1970 correlation",
    input_units=["dimensionless", "dimensionless", "m**2/s", "m"],
)
correlations.append(h_l_briggs)

E_g = Correlation(
    identifier="E_g",
    function=lambda diameter, u_g: get_E_g(
        diameter=diameter, u_g=u_g
    ),  # gas phase axial dispersion coefficient
    corr_type=CorrelationType.GAS_PHASE_DISPERSION,
    source="Malara 1995",
    description="gas phase axial dispersion coefficient [m2/s], Malara 1995 correlation models dispersion of the gas velocity distribution around the mean bubble velocity",
    input_units=["m", "m/s"],
)
correlations.append(E_g)


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


def get_u_g0(Eo, Mo, Re, mu_l, rho_l, d_b) -> float:  # TODO move inside class ?
    """
    bubble initial velocity [m/s], correlation for terminal velocity from Clift 1978
    """
    H = (4 / 3 * Eo.magnitude * Mo.magnitude**-0.149) * (
        mu_l.magnitude / 0.0009
    ) ** -0.14
    if H > 59.3:
        J = 3.42 * H**0.441
    elif H > 2:
        J = 0.94 * H**0.757
    else:
        J = Re * Mo**0.149 + 0.857
        warnings.warn(
            f"Warning: low Reynolds number {Re:.2e}, bubble size d_b = {d_b} m might be too small."
            f"Clift correlation will use default value for bubble velocity u_g0 = {U_G0_DEFAULT} m/s"
        )
    u_g0 = mu_l / (rho_l * d_b) * Mo**-0.149 * (J - 0.857)
    if u_g0 > ureg("1 m/s") or u_g0 < ureg("0.1 m/s"):
        warnings.warn(f"Warning: bubble velocity {u_g0} is out of the typical range")

    return u_g0


def get_eps_g(T, P_0, sigma_l, d_b, flow_g, tank_diameter, u_g0) -> float:
    eps_g = (
        const_R
        * T
        / (P_0 + 4 * sigma_l / d_b)
        * flow_g
        / (np.pi * (tank_diameter / 2) ** 2 * u_g0)
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
