"""Construct a standard input and visualize the dependency graph"""

from sparging import (
    LIBRA_PI_GEOM,
    BreederMaterial,
    LIBRA_PI_OPERATING_PARAMS,
    SpargingParameters,
    SimulationInput,
    all_correlations,
    VERBOSE_LEVEL,
)
import networkx as nx
from pyvis.network import Network
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=VERBOSE_LEVEL)

# construct helper objects
geom = LIBRA_PI_GEOM

flibe = BreederMaterial(
    name="FLiBe",
)

operating_params = LIBRA_PI_OPERATING_PARAMS

sparging_params = SpargingParameters(
    h_l=all_correlations("h_l_briggs"),
)

# construct input from helper objects
graph = nx.Graph()

my_input = SimulationInput.from_parameters(
    column_geometry=geom,
    breeder_material=flibe,
    operating_params=operating_params,
    sparging_params=sparging_params,
    graph=graph,
)

# print and visualize graph
for node in graph.nodes:
    graph.nodes[node]["value"] = str(graph.nodes[node]["value"])
    print(graph.nodes[node])
net = Network()
net.from_nx(graph)
net.show("model_dependencies.html", notebook=False)
