"""
libra_sparging: A finite element model for sparging processes using FEniCSx/DOLFINX.
"""

from .config import ureg, const_R, const_g, VERBOSE_LEVEL
from .ard_model import SimulationResults
from .simulation_input import (
    ColumnGeometry,
    BreederMaterial,
    OperatingParameters,
    SpargingParameters,
    SimulationInput,
)
from .ard_model import Simulation
from .animation import ConcentrationAnimator
from .correlations import (
    all_correlations,
    CorrelationGroup,
    Correlation,
    CorrelationType,
)
from .example_cases import (
    get_sim_input_LIBRA_Pi,
    get_sim_input_LIBRA1L,
    get_sim_input_standard,
    get_sim_input_malara,
    LIBRA_PI_GEOM,
    LIBRA_PI_MAT,
    LIBRA_PI_OPERATING_PARAMS,
    LIBRA_PI_SPARGING_PARAMS,
)

__all__ = [
    "SimulationInput",
    "SimulationResults",
    "ConcentrationAnimator",
    "ureg",
    "const_R",
    "const_g",
    "Simulation",
]
