from __future__ import annotations
from sparging.simulation_input import (
    ColumnGeometry,
    BreederMaterial,
    OperatingParameters,
    SpargingParameters,
    SimulationInput,
)
from sparging.config import ureg
from sparging.closure import all_closures
import logging
from typing import TYPE_CHECKING
import networkx as nx

if TYPE_CHECKING:
    import pint

logger = logging.getLogger(__name__)


def get_sim_input_LIBRA1L() -> tuple[SimulationInput, pint.Quantity]:
    """Returns the SimulationInput for the LIBRA 1L experiment, and the irradiation time (t_irr) to be used in the signal_irr function.
    The irradiation time is calculated based on the neutron fluence and generation rate reported in the LIBRA 1L paper."""
    geom = ColumnGeometry(
        area=170 * ureg.cm**2,
        height=7 * ureg.cm,
        nozzle_diameter=1.4 * ureg.mm,
        nb_nozzle=1 * ureg.dimensionless,
    )

    flibe = BreederMaterial(
        name="FLiBe",
    )

    operating_params = OperatingParameters(
        temperature=600 * ureg.celsius,
        P_top=1 * ureg.atm,
        ndot_g0=40 * ureg.sccm,
        tbr=2e-3 * ureg("triton / neutron"),  # according to LIBRA 1L paper
        n_gen_rate=1e9 * ureg("neutron / s"),
    )

    sparging_params = SpargingParameters(
        h_l=all_closures("h_l_briggs"),
    )

    libra_1L = SimulationInput.from_parameters(
        geom, flibe, operating_params, sparging_params
    )
    logger.info(libra_1L)

    n_fluence = 2.5e13 * ureg("neutron")
    n_gen_rate = operating_params.n_gen_rate
    t_irr = n_fluence / n_gen_rate

    libra_1L.signal_irr = lambda t: 1 if t <= t_irr else 0

    return (libra_1L, t_irr)


def get_sim_input_standard() -> SimulationInput:
    """Returns a standard SimulationInput that can be used for testing and tutorials.
    The parameters are not based on any specific experiment, but are chosen to be representative of a typical sparging system."""
    geom = ColumnGeometry(
        area=0.2 * ureg.m**2,
        height=1 * ureg.m,
        nozzle_diameter=0.001 * ureg.m,
        nb_nozzle=10 * ureg.dimensionless,
    )

    flibe = BreederMaterial(
        name="FLiBe",
    )

    operating_params = OperatingParameters(
        temperature=600 * ureg.celsius,
        P_top=1 * ureg.atm,
        ndot_g0=400 * ureg.sccm,
        tbr=0.1 * ureg("triton / neutron"),
        n_gen_rate=1e9 * ureg("neutron / s"),
    )

    sparging_params = SpargingParameters(
        h_l=all_closures("h_l_briggs"),
    )

    my_input = SimulationInput.from_parameters(
        geom, flibe, operating_params, sparging_params
    )
    logger.info(my_input)
    return my_input


def get_sim_input_LIBRA_Pi() -> SimulationInput:
    """
    LIBRA Pi (salt covers a solid angle of Pi with respect to the neutron source) is a quarter of the envisionned LIBRA One, with shape:
        - a quarter of a cylinder of radius 43 cm
        - with a central cylindrical hole of radius 15 cm to accomodate the neutron source
        - salt filled up to height of 93 cm
    -> salt volume = 0.93 * pi * (0.43^2 - 0.15^2) / 4 = 0.12 m^3
    -> equivalent base area if approximated as a cylinder = 0.12 m^3 / 0.93 m = 0.13 m^2
    (equivalent diameter = sqrt(4 * 0.13 / pi) = 0.41 m)

    - salt material: FLiBe, chosen as surrogate for actually employed CLiF with unknown properties
    - gas flow rate ~500 sccm limited by bubblers that strip tritium from the gas
    - sparger = 2 mm holes, 4 nozzles (to be designed, assume cross shaped sparger coming from the top)
    - pressure at the top = 1.2 atm (slightly higher than atmosphere because of sweep gas in the headspace)
    - temperature = 550 °C, limited by available heaters
    - tritium breeding: 2*10^9 neutrons/s (Starfire neutron source), TBR = 0.1 (calculated for this specific geometry)
    """
    geom = ColumnGeometry(
        area=0.13 * ureg.m**2,
        height=0.93 * ureg.m,
        nozzle_diameter=0.002 * ureg.m,
        nb_nozzle=4 * ureg.dimensionless,
    )

    flibe = BreederMaterial(
        name="FLiBe",
    )

    operating_params = OperatingParameters(
        temperature=550 * ureg.celsius,
        P_top=1.2 * ureg.atm,
        ndot_g0=500 * ureg.sccm,
        tbr=0.1 * ureg("triton / neutron"),
        n_gen_rate=2e9 * ureg("neutron / s"),
    )

    sparging_params = SpargingParameters(
        h_l=all_closures("h_l_briggs"),
    )

    my_input = SimulationInput.from_parameters(
        geom, flibe, operating_params, sparging_params
    )
    logger.info(my_input)
    return my_input


def get_sim_input_malara() -> SimulationInput:
    """
    To compare closure relations results with Malara 1995 paper
    """
    geom = ColumnGeometry(
        area=0.2 * ureg.m**2,
        height=3 * ureg.m,
        nozzle_diameter=0.001 * ureg.m,
        nb_nozzle=10 * ureg.dimensionless,
    )

    flibe = BreederMaterial(
        name="FLiBe",
    )

    operating_params = OperatingParameters(
        temperature=623 * ureg.kelvin,
        P_top=5e5 * ureg.pascal,
        ndot_g0=0.19 * ureg("mol / s"),
        tbr=0.1 * ureg("triton / neutron"),
        n_gen_rate=1e9 * ureg("neutron / s"),
    )

    sparging_params = SpargingParameters(
        h_l=all_closures("h_l_briggs"),
    )
    graph = nx.Graph()
    graph.add_node(
        "d_b", value=6 * ureg.mm, origin="input"
    )  # need a correlation for d_b and v_g adapted to such high gas flow rate
    my_input = SimulationInput.from_parameters(
        geom, flibe, operating_params, sparging_params, graph=graph
    )
    logger.info(my_input)
    return my_input


LIBRA_PI_GEOM = ColumnGeometry(
    area=0.13 * ureg.m**2,
    height=0.93 * ureg.m,
    nozzle_diameter=2 * ureg.mm,
    nb_nozzle=4 * ureg.dimensionless,
)

LIBRA_PI_MAT = BreederMaterial(
    name="FLiBe",
)

LIBRA_PI_OPERATING_PARAMS = OperatingParameters(
    temperature=550 * ureg.celsius,
    P_top=1.2 * ureg.atm,
    ndot_g0=500 * ureg.sccm,
    tbr=0.1 * ureg("triton / neutron"),
    n_gen_rate=2e9 * ureg("neutron / s"),
)

LIBRA_PI_SPARGING_PARAMS = SpargingParameters(
    h_l=all_closures("h_l_briggs"),
)
