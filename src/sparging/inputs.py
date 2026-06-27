from __future__ import annotations
from dataclasses import dataclass, replace
from sparging.correlations import Correlation, all_correlations
import pint
from typing import List
import inspect
import numpy as np
import logging
from sparging.config import ureg, const_R
from collections.abc import Callable
import networkx as nx


logger = logging.getLogger(__name__)


@dataclass
class ColumnGeometry:
    area: pint.Quantity
    height: pint.Quantity
    nozzle_diameter: pint.Quantity
    nb_nozzle: int

    @property
    def tank_diameter(self):
        return np.sqrt(4 * self.area / np.pi)

    @property
    def tank_volume(self):
        return self.area * self.height

    def copy(self):
        return replace(self)


@dataclass
class BreederMaterial:
    name: str
    D_l: pint.Quantity | Correlation | None = None
    K_s: pint.Quantity | Correlation | None = None
    density: pint.Quantity | Correlation | None = None
    viscosity: pint.Quantity | Correlation | None = None
    surface_tension: pint.Quantity | Correlation | None = None

    def copy(self):
        return replace(self)


@dataclass
class OperatingParameters:
    temperature: pint.Quantity
    flow_g_mol: pint.Quantity
    P_top: pint.Quantity
    # irradiation_signal: pint.Quantity # TODO implement
    # t_sparging: pint.Quantity # TODO implement
    flow_g_vol: pint.Quantity | None = None
    P_bottom: pint.Quantity | Correlation | None = None
    tbr: pint.Quantity | None = None
    n_gen_rate: pint.Quantity | None = None
    Q_T: pint.Quantity | None = None

    def copy(self):
        return replace(self)


@dataclass
class SpargingParameters:
    h_l: pint.Quantity | Correlation
    eps_g: pint.Quantity | Correlation | None = None
    u_g0: pint.Quantity | Correlation | None = None
    d_b: pint.Quantity | Correlation | None = None
    rho_g: pint.Quantity | Correlation | None = None
    E_g: pint.Quantity | Correlation | None = None
    E_l: pint.Quantity | Correlation | None = None
    a: pint.Quantity | Correlation | None = None

    def copy(self):
        return replace(self)


@dataclass  # should it not be a normal class ? (init method)s
class SimulationInput:
    height: pint.Quantity
    area: pint.Quantity
    u_g0: pint.Quantity
    temperature: pint.Quantity
    a: pint.Quantity
    h_l: pint.Quantity
    K_s: pint.Quantity
    P_bottom: pint.Quantity
    rho_l: pint.Quantity
    eps_g: pint.Quantity
    E_g: pint.Quantity
    E_l: pint.Quantity
    Q_T: pint.Quantity
    signal_irr: Callable[[pint.Quantity], float] = lambda t: 1
    """callable = f:R+ (time) -> [0,1] """
    signal_sparging: Callable[[pint.Quantity], float] = lambda t: 1
    """callable = f:R+ (time) -> [0,1] """
    profile_source_T: Callable[[float], float] | None = None
    """callable = f:[0,1] -> R+, it takes a dimensionless coordinate: (z / height)"""
    c_T2_0: pint.Quantity = 0 * ureg("molT2/m**3")
    profile_c_T2_0: Callable[[float], pint.Quantity] | None = None
    """callable = f:[0,1] -> R+, it takes a dimensionless coordinate: (z / height)"""
    required_keys = (
        "height",
        "area",
        "u_g0",
        "temperature",
        "a",
        "h_l",
        "K_s",
        "P_bottom",
        "rho_l",
        "eps_g",
        "E_g",
        "E_l",
        "Q_T",
    )  # these parameters will be used to solve the model
    graph: nx.Graph | None = None
    """ Stores the intermediate parameters and their relationships that built the SimulationInput. 
    It has attributes "nodes" and "edges".
    Each node is accessed by its ID (the param name) and contains a dictionary with properties:
        - value: value of the parameter as pint.Quantity
        - origin: "input" | correlation identifier
    e.g use: mySimulationInput.graph.nodes["height"]["value"]
    """

    @property
    def volume(self):
        return self.area * self.height

    @property
    def eps_l(self):
        return 1 - self.eps_g

    def set_S_T(self, val: pint.Quantity):
        self.Q_T = (val.to("molT/m**3/s") * self.volume).to("molT/s")

    def get_S_T(self) -> pint.Quantity:
        return (self.Q_T / self.volume).to("molT/m**3/s")

    def get_tau(self) -> pint.Quantity:
        """characteristic time of the sparger under the small partial pressure (SPP) approximation"""
        return (self.eps_l / (self.h_l * self.a)).to("seconds")

    def get_c_T2_SS(self) -> pint.Quantity:
        return (self.get_S_T() * 1 / (self.h_l * self.a)).to("molT2/m^3")

    def get_Pi_number(self) -> pint.Quantity:
        """Partial pressure number,
        If Pi < ~0.1, then gas partial pressure can be neglected in front of liquid concentration, we are in the small partial pressure (SPP) regime,
        If Pi > ~0.1, then partial pressure starts limiting the interfacial mass transfer (PPL regime)
        """
        return (
            self.K_s
            * (const_R * self.temperature)
            * self.height
            * self.h_l
            * self.a
            / (self.eps_g * self.u_g0)
        ).to("dimensionless")

    def get_dP_dx(self) -> pint.Quantity:
        """
        returns 1/c_T * dP_T/dx in the SPP approximation (linearized around P_T = 0))
        """
        return (
            6
            * (const_R * self.temperature)
            * self.h_l
            / (self.graph.nodes["d_b"]["value"] * self.u_g0)
        ).to("Pa/(mol/m^3)/m")

    def __post_init__(self):
        # make sure there are only pint.Quantity or callables in the input, otherwise raise an error
        for key in self.required_keys:
            value = getattr(self, key)
            if not isinstance(value, pint.Quantity):
                raise ValueError(
                    f"In {self.__class__.__name__}: Invalid type for '{key}': expected a pint.Quantity, got {value} of type {type(value)}"
                )

    def to_json(self, path: str):
        import json

        with open(path, "w") as f:
            json.dump(
                {
                    key: str(value)
                    for key, value in self.__dict__.items()
                    if value is not None and not callable(value)
                },
                f,
                indent=2,
            )

    @classmethod
    def from_parameters(
        cls,
        column_geometry: ColumnGeometry,
        breeder_material: BreederMaterial,
        operating_params: OperatingParameters,
        sparging_params: SpargingParameters,
        graph: nx.Graph | None = None,
    ):
        """
        - graph: optional, if want to get the values of intermediate parameters and visualize how the input was constructed. Each node has an ID (the param name) and contains a dictionary with properties:
            - value: value of the parameter as pint.Quantity
            - origin: "input" | correlation identifier
        """
        input_objects = [
            column_geometry,
            breeder_material,
            operating_params,
            sparging_params,
        ]
        discovered_graph = nx.Graph() if graph is None else graph

        for required_key in cls.required_keys:
            find_in_graph(required_key, discovered_graph, input_objs=input_objects)

        return cls(
            graph=discovered_graph,
            **{arg: discovered_graph.nodes[arg]["value"] for arg in cls.required_keys},
        )

    def __str__(self):
        return "\n\t".join(
            [
                f"{name}: {value}"
                for name in self.__dataclass_fields__
                for value in [getattr(self, name)]
            ]
        )

    def copy(self):
        return replace(self)


def find_in_graph(
    required_node: str,
    discovered_graph: nx.Graph,
    input_objs: List[
        SpargingParameters | OperatingParameters | BreederMaterial | ColumnGeometry
    ],
) -> None:
    """Abstracts SimulationInput construction as a graph search problem. "Correlation" object are seen as a path to the corresponding node
    - required_node: parameter we want to obtain (e.g. h_l)
    - discovered_graph: already discovered parameters
        - graph properties:
            - id : name of the parameter
            - value: value of the parameter as pint.Quantity
            - origin: "input" | <correlation identifier>
    - input_objs: list of objects in which to search
    - returns the updated discovered_nodes with the required_node added
    """
    # first check if the required node is already discovered
    if required_node in discovered_graph.nodes:
        logger.verbose(f"Found required node '{required_node}' in discovered nodes...")
        return

    # then check if the required node is given as input (either as a pint.Quantity or as a Correlation)
    if (result := check_input(required_node, input_objs)) is None:
        # if it's not, look for default correlation
        if required_node in all_correlations:
            result = all_correlations(required_node)
            logger.verbose(
                f"Found default correlation for required node '{required_node}': {result.identifier}"
            )
        else:
            raise ValueError(
                f"Could not find path to required node '{required_node}' in the graph or in the default correlations"
            )
    if isinstance(result, pint.Quantity):
        # value was given as input
        discovered_graph.add_node(required_node, value=result, origin="input")
    elif isinstance(result, Correlation):
        # no value specified, get value from correlation
        discovered_graph.add_node(required_node, origin=result.identifier)
        result = resolve_correlation(
            node_id=required_node,
            corr=result,
            discovered_graph=discovered_graph,
            input_objs=input_objs,
        )  # also update discovered_graph with the nodes possibly discovered during recursive search
        discovered_graph.nodes[required_node]["value"] = result

    assert isinstance(result, pint.Quantity), (
        f"Result for required node '{required_node}' is not a pint.Quantity after resolution, got {result} of type {type(result)}"
    )


def check_input(
    required_node: str,
    input_objs: List[
        SpargingParameters | OperatingParameters | BreederMaterial | ColumnGeometry
    ],
) -> pint.Quantity | Correlation | None:
    """look for pint.Quantity or Correlation given in input objects"""
    result = None
    for object in input_objs:
        # scan for the required node in the attributes of the object
        if (result := getattr(object, required_node, None)) is not None:
            if isinstance(result, pint.Quantity):
                # required node was found
                logger.verbose(
                    f"Found Quantity for required node '{required_node}' in input: {result}"
                )
                break
            elif isinstance(result, Correlation):
                logger.verbose(
                    f"Found correlation for required node '{required_node}' in input: {result.identifier}"
                )
                break
            else:
                raise ValueError(
                    f"In check_input: found result for '{required_node}': but expected a Correlation or a pint.Quantity, got {result} of type {type(result)}"
                )
    return result


def resolve_correlation(
    node_id: str,
    corr: Correlation,
    discovered_graph: nx.Graph,
    input_objs: List[
        SpargingParameters | OperatingParameters | BreederMaterial | ColumnGeometry
    ],
) -> pint.Quantity:
    """Recursively resolve a correlation by first resolving its arguments, then applying the correlation function to the resolved arguments.
    - corr: Correlation object to resolve
    - discovered_graph: graph containing already resolved quantities, to avoid redundant calculations and infinite recursion
    - input_objs: list of objects in which to search for the arguments of the correlation
    - returns the resolved value of the correlation as a pint.Quantity"""
    corr_args = inspect.signature(corr.function).parameters.keys()
    for arg in corr_args:
        logger.verbose(
            f"Resolving argument '{arg}' for correlation '{corr.identifier}'..."
        )
        find_in_graph(arg, discovered_graph, input_objs)
        discovered_graph.add_edge(node_id, arg)

    assert all(arg in discovered_graph.nodes for arg in corr_args), (
        f"Could not resolve all arguments for correlation '{corr.identifier}'. "
        f"Missing arguments: {[arg for arg in corr_args if arg not in discovered_graph.nodes]}"
    )

    return corr(**{arg: discovered_graph.nodes[arg]["value"] for arg in corr_args})
