from sparging.inputs import (
    ColumnGeometry,
    BreederMaterial,
    OperatingParameters,
    SpargingParameters,
    find_in_graph,
    check_input,
    SimulationInput,
)
from sparging import all_correlations, ureg
from sparging.config import VERBOSE_LEVEL
import networkx as nx
from pyvis.network import Network
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=VERBOSE_LEVEL)

geom = ColumnGeometry(
    area=0.2 * ureg.m**2,
    height=1.0 * ureg.m,
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

graph = nx.Graph()
# find_in_graph("drho", graph, [geom, flibe, operating_params, sparging_params])

# print("Nodes in graph:")
# for node in graph.nodes(data=True):
#     print(node)

my_input = SimulationInput.from_parameters(
    column_geometry=geom,
    breeder_material=flibe,
    operating_params=operating_params,
    sparging_params=sparging_params,
    graph=graph,
)

for node in graph.nodes:
    graph.nodes[node]["value"] = str(graph.nodes[node]["value"])
    print(graph.nodes[node])
net = Network()
net.from_nx(graph)
net.show("model_dependencies.html", notebook=False)
