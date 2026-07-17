from __future__ import annotations
from sparging.config import ureg, const_R, const_g
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparging.model import SimulationInput
    import pint
import numpy as np
import scipy.constants as const
import warnings

from dataclasses import dataclass
import enum

PROFILE = "profile"


class CorrelationType(enum.Enum):  # TODO do we really use it ?
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
    SUPERFICIAL_GAS_VELOCITY = "u_g"
    BUBBLE_VELOCITY = "v_g0"
    GAS_PHASE_DISPERSION = "E_g"
    LIQUID_PHASE_DISPERSION = "E_l"
    FLOW_RATE = "ndot_g0"
    INTERFACIAL_AREA = "a"
    TRITIUM_SOURCE = "source_T"
    LIQUID_PRESSURE_PROFILE = "P_l"
    GAS_PRESSURE_PROFILE = "P_g"


@dataclass
class Correlation:
    identifier: str
    function: callable
    corr_type: CorrelationType
    input_units: list[str]
    source: str | None = None
    description: str | None = None
    output_units: str | None = None

    def _validate_inputs(self, kwargs: dict[str, pint.Quantity]) -> None:
        for arg_name, expected in zip(kwargs, self.input_units):
            arg = kwargs[arg_name]
            if expected == PROFILE:
                if not callable(arg):
                    raise ValueError(
                        f"{self.identifier}: argument '{arg_name}' expected to be a "
                        f"profile (callable f(z) -> Quantity), got {type(arg)}"
                    )
                continue
            if not isinstance(arg, ureg.Quantity):
                raise ValueError(
                    f"Invalid input: expected a pint.Quantity with units of "
                    f"{expected}, got {arg} of type {type(arg)}"
                )
            if arg.dimensionality != ureg(expected).dimensionality:
                raise ValueError(
                    f"Invalid input when resolving for {self.identifier}: expected "
                    f"dimensions of {expected}, got {arg.dimensionality}"
                )

    def __call__(self, **kwargs: pint.Quantity) -> pint.Quantity:
        self._validate_inputs(kwargs)
        result = self.function(**kwargs)
        if self.output_units is not None:
            return result.to(self.output_units)
        else:
            return result.to_base_units()

    # TODO add a method that checks the validity of the input parameters based on the range of validity of the correlation, if provided in the description or source. This method could be called before running the simulation to warn the user if they are using a correlation outside of its validated range.
    # TODO add __post_init__ to check that user defined correlation has same number of input_units as the number of arguments in the function, and that the output of the function is a pint.Quantity with the correct units if output_units is provided


@dataclass
class Profile(Correlation):
    """A closure relation that resolves to a *spatial profile*: a callable
    f(z) -> pint.Quantity, instead of a scalar pint.Quantity.

    `function(**inputs)` must return a callable mapping a position (length
    Quantity) to a Quantity expressed in `output_units`.
    """

    def __call__(self, **kwargs: pint.Quantity):
        self._validate_inputs(kwargs)
        profile_func = self.function(**kwargs)
        if not callable(profile_func):
            raise ValueError(
                f"Profile '{self.identifier}' must return a callable, "
                f"got {type(profile_func)}"
            )
        if self.output_units is None:
            return profile_func
        # wrap so the profile always yields output_units
        return lambda z, _f=profile_func: _f(z).to(self.output_units)


class CorrelationGroup(list[Correlation]):
    def __call__(self, identifier: str) -> Correlation:
        for corr in self:
            if corr.identifier == identifier:
                return corr
        raise ValueError(
            f"Correlation with identifier {identifier} not found in correlation group"
        )

    def __contains__(self, key: str | Correlation):
        if isinstance(key, str):
            for corr in self:
                if corr.identifier == key:
                    return True
            return False
        elif isinstance(key, Correlation):
            return super().__contains__(key)
        else:
            raise TypeError(
                f"Type not valid for correlation group membership check, expected str or Correlation, got {type(key)}"
            )

    def get_list(self, corr_type: CorrelationType) -> list[Correlation]:
        return [corr for corr in self if corr.corr_type == corr_type]


all_correlations = CorrelationGroup([])


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
all_correlations.append(rho_l)

mu_l = Correlation(
    identifier="mu_l",
    function=lambda temperature: ureg.Quantity(
        0.116e-3 * np.exp(3755 / temperature.to("kelvin").magnitude), "Pa*s"
    ),  # dynamic viscosity of Li2BeF4, Cantor 1968
    corr_type=CorrelationType.VISCOSITY,
    source="Cantor 1968",
    description="dynamic viscosity of Li2BeF4 as a function of temperature",
    input_units=["kelvin"],
)
all_correlations.append(mu_l)

nu_l = Correlation(
    identifier="nu_l",
    function=lambda mu_l, rho_l: (
        mu_l / rho_l
    ),  # kinematic viscosity of Li2BeF4 calculated from dynamic viscosity and density
    corr_type=CorrelationType.VISCOSITY,
    source="calculated from mu_l and rho_l",
    description="kinematic viscosity of Li2BeF4 calculated from dynamic viscosity and density",
    input_units=["Pa*s", "kg/m**3"],
    output_units="m**2/s",
)
all_correlations.append(nu_l)

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
all_correlations.append(sigma_l)

# TODO this could leverage HTM
D_l_calderoni = Correlation(
    identifier="D_l_calderoni",
    function=lambda temperature: (
        9.3e-7
        * ureg("m**2/s")
        * np.exp(-42e3 * ureg("J/mol") / (const_R * temperature.to("kelvin")))
    ),
    corr_type=CorrelationType.DIFFUSIVITY,
    source="Calderoni 2008",
    description="diffusivity of tritium in liquid FLiBe as a function of temperature",
    input_units=["kelvin"],
    output_units="m**2/s",
)
all_correlations.append(D_l_calderoni)

D_l_oishi = Correlation(
    identifier="D_l_oishi",
    function=lambda temperature: (
        7.57e-7
        * ureg("m**2/s")
        * np.exp(-36.7e3 * ureg("J/mol") / (const_R * temperature.to("kelvin")))
    ),
    corr_type=CorrelationType.DIFFUSIVITY,
    source="Oishi 1989",
    description="diffusivity of tritium in liquid FLiBe as a function of temperature",
    input_units=["kelvin"],
    output_units="m**2/s",
)
all_correlations.append(D_l_oishi)

D_l = D_l_calderoni  # default diffusivity correlation, can be overridden by user defined correlation
D_l.identifier = "D_l"
all_correlations.append(D_l)

K_s = Correlation(
    identifier="K_s",
    function=lambda temperature: (
        7.9e-2
        * ureg("mol/m**3/Pa")
        * np.exp(-35e3 * ureg("J/mol") / (const_R * temperature.to("kelvin")))
    ),
    corr_type=CorrelationType.SOLUBILITY,
    source="Calderoni 2008",
    description="solubility of tritium in liquid FLiBe as a function of temperature",
    input_units=["kelvin"],
    output_units="mol/m**3/Pa",
)
all_correlations.append(K_s)

d_b0 = Correlation(
    identifier="d_b0",
    function=lambda Vdot_g0, nozzle_diameter, nb_nozzle: get_d_b0(
        Vdot_g0=Vdot_g0, nozzle_diameter=nozzle_diameter, nb_nozzle=nb_nozzle
    ),  # mean bubble diameter, Kanai 2017
    corr_type=CorrelationType.BUBBLE_DIAMETER,
    input_units=["m**3/s", "m", "dimensionless"],
    output_units="m",
    source="Kanai 2017 (https://doi.org/10.1252/jcej.15we307); report by Evans 2026 (https://doi.org/10.1016/j.nucengdes.2025.114624)",
    description="Mean bubble diameter, validated for nitrogen sparging in NaNO3 molten salt at 643K and gas flow rates of 3-10 cm3/s. Author suggests it may be applicable to FLiNaK and FLiBe.",
)
all_correlations.append(d_b0)

Eo = Correlation(
    identifier="Eo",
    function=lambda drho, d_b0, sigma_l: (const_g * drho * d_b0**2 / sigma_l).to(
        "dimensionless"
    ),  # Eotvos number
    corr_type=CorrelationType.EOTVOS_NUMBER,
    input_units=["kg/m**3", "m", "N/m"],
    output_units="dimensionless",
)
all_correlations.append(Eo)

Mo = Correlation(
    identifier="Mo",
    function=lambda drho, mu_l, rho_l, sigma_l: (
        drho * const_g * mu_l**4 / (rho_l**2 * sigma_l**3)
    ).to("dimensionless"),  # Morton number
    corr_type=CorrelationType.MORTON_NUMBER,
    input_units=["kg/m**3", "Pa*s", "kg/m**3", "N/m"],
    output_units="dimensionless",
)
all_correlations.append(Mo)

Sc = Correlation(
    identifier="Sc",
    function=lambda nu_l, D_l: (nu_l / D_l).to("dimensionless"),  # Schmidt number
    corr_type=CorrelationType.SCHMIDT_NUMBER,
    input_units=["m**2/s", "m**2/s"],
    output_units="dimensionless",
)
all_correlations.append(Sc)

# Bubble Reynolds number
Re = Correlation(
    identifier="Re",
    function=lambda rho_l, v_g0, d_b0, mu_l: (rho_l * v_g0 * d_b0 / mu_l).to(
        "dimensionless"
    ),
    corr_type=CorrelationType.REYNOLDS_NUMBER,
    input_units=["kg/m**3", "m/s", "m", "Pa*s"],
    output_units="dimensionless",
)
all_correlations.append(Re)

v_g0 = Correlation(
    identifier="v_g0",
    function=lambda Eo, Mo, mu_l, rho_l, d_b0: get_v_g0(
        Eo=Eo, Mo=Mo, mu_l=mu_l, rho_l=rho_l, d_b=d_b0
    ),  # initial gas velocity
    corr_type=CorrelationType.BUBBLE_VELOCITY,
    input_units=[
        "dimensionless",
        "dimensionless",
        "Pa*s",
        "kg/m**3",
        "m",
    ],
    output_units="m/s",
    source="Chavez 2021: https://doi.org/10.1016/j.ijheatfluidflow.2021.108875",
    description="Clift 1978 correlation for terminal velocity, validated for single He bubble rising in steady FLiNaK. Likely to be applicable to FLiBe (similar surface tensions, density and viscosity).",
)
all_correlations.append(v_g0)


h_l_higbie = Profile(
    identifier="h_l_higbie",
    function=lambda D_l, v_g0, d_b: (
        lambda z: get_h_higbie(D_l=D_l, v_g=v_g0, d_b=d_b(z))
    ),  # mass transfer coefficient profile with Higbie correlation
    corr_type=CorrelationType.MASS_TRANSFER_COEFF,
    source="Higbie 1935",
    description="mass transfer coefficient profile for tritium in liquid FLiBe using Higbie penetration model; varies with height through the bubble diameter profile d_b(z)",
    input_units=["m**2/s", "m/s", PROFILE],
    output_units="m/s",
)
all_correlations.append(h_l_higbie)


h_l_briggs = Profile(
    identifier="h_l_briggs",
    function=lambda rho_l, v_g0, mu_l, Sc, D_l, d_b: (
        lambda z: get_h_briggs(
            Re=(rho_l * v_g0 * d_b(z) / mu_l).to("dimensionless"),
            Sc=Sc,
            D_l=D_l,
            d_b=d_b(z),
        )
    ),  # mass transfer coefficient profile with Briggs correlation
    corr_type=CorrelationType.MASS_TRANSFER_COEFF,
    source="Briggs 1970",
    description="mass transfer coefficient profile for tritium in liquid FLiBe using Briggs 1970 correlation; varies with height through the bubble diameter profile d_b(z) (Re is recomputed locally from d_b(z))",
    input_units=["kg/m**3", "m/s", "Pa*s", "dimensionless", "m**2/s", PROFILE],
    output_units="m/s",
)
all_correlations.append(h_l_briggs)

# liquid phase axial dispersion coefficient
E_l = Correlation(
    identifier="E_l",
    function=lambda tank_diameter, u_g: ureg.Quantity(
        0.678 * tank_diameter.magnitude**1.4 * u_g(0 * ureg.m).magnitude ** 0.3,
        "m**2/s",
    ),
    corr_type=CorrelationType.LIQUID_PHASE_DISPERSION,
    source="Deckwer 1974",
    description="liquid phase axial dispersion coefficient",
    input_units=["m", PROFILE],
    output_units="m**2/s",
)
all_correlations.append(E_l)

# gas phase axial dispersion coefficient
E_g = Correlation(
    identifier="E_g",
    function=lambda tank_diameter, u_g: (
        0.2 * ureg("1/m") * tank_diameter**2 * u_g(0 * ureg.m)
    ),  # gas phase axial dispersion coefficient
    corr_type=CorrelationType.GAS_PHASE_DISPERSION,
    source="Malara 1995",
    description="gas phase axial dispersion coefficient [m2/s], Malara 1995",
    input_units=["m", PROFILE],
    output_units="m**2/s",
)
all_correlations.append(E_g)

drho = Correlation(
    identifier="drho",
    function=lambda rho_l, rho_g: (rho_l - rho_g).to(
        "kg/m**3"
    ),  # density difference between liquid and gas
    corr_type=CorrelationType.DENSITY,
    input_units=["kg/m**3", "kg/m**3"],
)
all_correlations.append(drho)


he_molar_mass = ureg("4.003e-3 kg/mol")
rho_g = Correlation(
    identifier="rho_g",
    function=lambda temperature, P_l: ureg.Quantity(
        (P_l(0 * ureg.m) * he_molar_mass / (const_R * temperature.to("kelvin"))).to(
            "kg/m**3"
        )
    ),  # ideal gas law for density of gas phase
    corr_type=CorrelationType.DENSITY,
    description="density of gas phase calculated using ideal gas law",
    input_units=["kelvin", PROFILE],
)
all_correlations.append(rho_g)


Vdot_g0 = Correlation(
    identifier="Vdot_g0",
    function=lambda ndot_g0, temperature, P_l: (
        ndot_g0 * const_R * temperature / P_l(0 * ureg.m)
    ),  # convert molar flow rate to volumetric flow rate using ideal gas law
    corr_type=CorrelationType.FLOW_RATE,
    description="volumetric flow rate of gas phase calculated from molar flow rate using ideal gas law",
    input_units=["mol/s", "kelvin", PROFILE],
    output_units="m**3/s",
)
all_correlations.append(Vdot_g0)


source_T_integral = Correlation(
    identifier="Q_T",
    function=lambda tbr, n_gen_rate: (
        tbr * n_gen_rate
    ),  # source term for tritium generation calculated from TBR and neutron generation rate
    corr_type=CorrelationType.TRITIUM_SOURCE,
    input_units=["triton/neutron", "neutron/s"],
    output_units="molT/s",
)
all_correlations.append(source_T_integral)


def get_d_b0(
    Vdot_g0: pint.Quantity, nozzle_diameter: pint.Quantity, nb_nozzle: pint.Quantity
) -> float:
    """
    mean bubble diameter [m], Kanai 2017 (reported by Evans 2026)
    """
    nozzle_flow = Vdot_g0 / nb_nozzle  # volumetric flow per nozzle [m3/s]
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


def get_v_g0(Eo, Mo, mu_l, rho_l, d_b) -> float:  # TODO move inside class ?
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
        raise ValueError(
            f"Clift correlation is not valid for H = {H}, which is calculated based on the input parameters. Check the input parameters and the validity of the correlation for the given range of parameters."
        )
    v_g0 = mu_l / (rho_l * d_b) * Mo**-0.149 * (J - 0.857)
    if v_g0 > ureg("1 m/s") or v_g0 < ureg("0.1 m/s"):
        warnings.warn(
            f"Warning: bubble terminal velocity {v_g0} is out of the typical range"
        )

    return v_g0


def get_h_higbie(D_l: float, v_g: float, d_b: float) -> float:
    """
    Higbie penetration model average mass transfer coefficient [m/s]-> suited for large mobile interfaces
    """
    h_l = 2 * ((D_l * v_g) / (const.pi * d_b)) ** 0.5
    return h_l


def get_h_briggs(Re: float, Sc: float, D_l: float, d_b: float) -> float:
    """
    Sherwood based mass transfer coefficient [m/s] for tritium in liquid FLiBe (Briggs 1970 correlation) -> suited for small rigid interfaces
    """
    Sh = 0.089 * Re**0.69 * Sc**0.33  # Sherwood number
    h_l = Sh * D_l / d_b
    return h_l


# hydrostatic pressure profile along tank height
P_l = Profile(
    identifier="P_l",
    function=lambda P_top, rho_l, height: (
        lambda z: P_top + rho_l * const_g * (height - z)
    ),
    corr_type=CorrelationType.LIQUID_PRESSURE_PROFILE,
    input_units=["Pa", "kg/m^3", "m"],
    output_units="Pa",
    description="hydrostatic pressure profile along tank height",
)
all_correlations.append(P_l)


P_g = Profile(
    identifier="P_g",
    function=lambda P_l, d_b, sigma_l: lambda z: P_l(z) + 4 * sigma_l / d_b(z),
    corr_type=CorrelationType.GAS_PRESSURE_PROFILE,
    input_units=[PROFILE, PROFILE, "N/m"],
    output_units="Pa",
    description="pressure in a mechanically stable bubble immerged in a liquid",
)
all_correlations.append(P_g)


d_b = Profile(
    identifier="d_b",
    function=lambda d_b0, P_l: lambda z: d_b0 * (P_l(0 * ureg.m) / P_l(z)) ** (1 / 3),
    corr_type=CorrelationType.BUBBLE_DIAMETER,
    input_units=["m", PROFILE],
    output_units="m",
    description="Bubble diameter profile from hydrostatic expansion",
)
all_correlations.append(d_b)


eps_g = Profile(
    identifier="eps_g",
    function=lambda u_g, v_g0: lambda z: u_g(z) / v_g0,
    corr_type=CorrelationType.GAS_VOID_FRACTION,
    input_units=[PROFILE, "m/s"],
    output_units="dimensionless",
    description="gas void fraction profile (local P and d_b)",
)
all_correlations.append(eps_g)

a = Profile(
    identifier="a",
    function=lambda eps_g, d_b: lambda z: 6 * eps_g(z) / d_b(z),
    corr_type=CorrelationType.INTERFACIAL_AREA,
    input_units=[PROFILE, PROFILE],
    output_units="1/m",
    description="specific interfacial area profile",
)
all_correlations.append(a)

u_g = Profile(
    identifier="u_g",
    function=lambda Vdot_g0, area, P_g: (
        lambda z: (Vdot_g0 / area).to("m/s") * (P_g(0 * ureg.m) / P_g(z))
    ),
    corr_type=CorrelationType.SUPERFICIAL_GAS_VELOCITY,
    input_units=["m^3/s", "m^2", PROFILE],
    output_units="m/s",
    description="superficial gas velocity profile",
)
all_correlations.append(u_g)
