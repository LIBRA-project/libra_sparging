"""
libra_sparging: A finite element model for sparging processes using FEniCSx/DOLFINX.
"""

from .config import ureg, const_R, const_g, VERBOSE_LEVEL
from .model import SimulationResults
from .inputs import (
    ColumnGeometry,
    BreederMaterial,
    OperatingParameters,
    SpargingParameters,
    SimulationInput,
)
from .model import Simulation
from .animation import ConcentrationAnimator
from .correlations import (
    all_correlations,
    CorrelationGroup,
    Correlation,
    CorrelationType,
)
from .input_examples import (
    get_sim_input_LIBRA1L,
    get_sim_input_standard,
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
