from __future__ import annotations
from sparging.inputs import (
    ColumnGeometry,
    BreederMaterial,
    OperatingParameters,
    SpargingParameters,
    SimulationInput,
)
from sparging.config import ureg
from sparging.correlations import all_correlations
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
        flow_g_mol=40 * ureg.sccm,
        tbr=2e-3 * ureg("triton / neutron"),  # according to LIBRA 1L paper
        n_gen_rate=1e9 * ureg("neutron / s"),
    )

    sparging_params = SpargingParameters(
        h_l=all_correlations("h_l_briggs"),
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
        flow_g_mol=400 * ureg.sccm,
        tbr=0.1 * ureg("triton / neutron"),
        n_gen_rate=1e9 * ureg("neutron / s"),
    )

    sparging_params = SpargingParameters(
        h_l=all_correlations("h_l_briggs"),
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
        flow_g_mol=0.19 * ureg("mol / s"),
        tbr=0.1 * ureg("triton / neutron"),
        n_gen_rate=1e9 * ureg("neutron / s"),
    )

    sparging_params = SpargingParameters(
        h_l=all_correlations("h_l_briggs"),
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
    area=0.2 * ureg.m**2,
    height=1 * ureg.m,
    nozzle_diameter=1.5 * ureg.mm,
    nb_nozzle=5 * ureg.dimensionless,
)

LIBRA_PI_MAT = BreederMaterial(
    name="FLiBe",
)

LIBRA_PI_OPERATING_PARAMS = OperatingParameters(
    temperature=550 * ureg.celsius,
    P_top=1.2 * ureg.atm,
    flow_g_mol=400 * ureg.sccm,
    tbr=0.1 * ureg("triton / neutron"),
    n_gen_rate=1e9 * ureg("neutron / s"),
)

LIBRA_PI_SPARGING_PARAMS = SpargingParameters(
    h_l=all_correlations("h_l_briggs"),
)
