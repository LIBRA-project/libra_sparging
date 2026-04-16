from dataclasses import dataclass
from sparging.correlations import Correlation, all_correlations
import pint
from typing import List
import inspect
import numpy as np
import logging
from sparging.config import const_R

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


@dataclass
class BreederMaterial:
    name: str
    D_l: pint.Quantity | Correlation | None = None
    K_s: pint.Quantity | Correlation | None = None
    density: pint.Quantity | Correlation | None = None
    viscosity: pint.Quantity | Correlation | None = None
    surface_tension: pint.Quantity | Correlation | None = None


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


@dataclass
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
    D_l: pint.Quantity
    Q_T: pint.Quantity
    # normalize_source_T: bool = True

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

    def get_PP_number(self) -> pint.Quantity:
        """Partial pressure number, ratio of the equivalent T concentration at liquid boundary to the bulk liquid concentration.
        If PP << 1, then we are in the small partial pressure (SPP) regime, and if PP ~= 1, then we are in the partial pressure limited (PPL) regime.
        """
        return (
            self.K_s
            * (const_R * self.temperature)
            * self.height
            / (self.get_tau() * self.u_g0)
        ).to("dimensionless")

    def __post_init__(self):
        # make sure there are only pint.Quantity or callables in the input, otherwise raise an error
        for key in self.__dataclass_fields__.keys():
            value = getattr(self, key)
            if not isinstance(value, pint.Quantity):
                raise ValueError(
                    f"In {self.__class__.__name__}: Invalid type for '{key}': expected a pint.Quantity, got {value} of type {type(value)}"
                )

    def to_json(self, path: str):
        import json

        with open(path, "w") as f:
            json.dump(
                {key: str(value) for key, value in self.__dict__.items()}, f, indent=2
            )

    @classmethod
    def from_parameters(
        cls,
        column_geometry: ColumnGeometry,
        breeder_material: BreederMaterial,
        operating_params: OperatingParameters,
        sparging_params: SpargingParameters,
    ):
        input_objects = [
            column_geometry,
            breeder_material,
            operating_params,
            sparging_params,
        ]
        resolved_parameters = {}
        required_keys = (
            cls.__dataclass_fields__.keys()
        )  # these parameters will be used to solve the model

        for required_key in required_keys:
            find_in_graph(required_key, resolved_parameters, graph=input_objects)

        return cls(**{arg: resolved_parameters[arg] for arg in required_keys})

    def __str__(self):
        return "\n\t".join(
            [
                f"{name}: {value}"
                for name in self.__dataclass_fields__
                for value in [getattr(self, name)]
            ]
        )


def find_in_graph(
    required_node: str,
    discovered_nodes: dict,
    graph: List[
        SpargingParameters | OperatingParameters | BreederMaterial | ColumnGeometry
    ],
) -> None:
    """Abstracts SimulationInput construction as a graph search problem. "Correlation" object are seen as a path to the corresponding node
    - required_node: parameter we want to obtain (e.g. h_l)
    - discovered_nodes: already discovered parameters as pint.Quantity
    - graph: list of objects in which to search
    - returns the updated discovered_nodes with the required_node added
    """
    # first check if the required node is already discovered
    if required_node in discovered_nodes:
        logger.info(f"Found required node '{required_node}' in discovered nodes...")
        return

    # then check if the required node is given as input (either as a pint.Quantity or as a Correlation)
    if (result := check_input(required_node, graph)) is None:
        # if it's not, look for default correlation
        if required_node in all_correlations:
            result = all_correlations(required_node)
            logger.info(
                f"Found default correlation for required node '{required_node}': {result.identifier}"
            )
        else:
            raise ValueError(
                f"Could not find path to required node '{required_node}' in the graph or in the default correlations"
            )
    if isinstance(result, Correlation):
        result = resolve_correlation(
            corr=result, resolved_quantities=discovered_nodes, graph=graph
        )  # also update discovered_nodes with the nodes possibly discovered during recursive search

    assert isinstance(result, pint.Quantity), (
        f"Result for required node '{required_node}' is not a pint.Quantity after resolution, got {result} of type {type(result)}"
    )
    discovered_nodes.update({required_node: result})


def check_input(
    required_node: str, input_objs: list[object]
) -> pint.Quantity | Correlation | None:
    """look for pint.Quantity or Correlation given in input objects"""
    result = None
    for object in input_objs:
        # scan for the required node in the attributes of the object
        if (result := getattr(object, required_node, None)) is not None:
            if isinstance(result, pint.Quantity):
                # required node was found
                logger.info(
                    f"Found Quantity for required node '{required_node}' in graph: {result}"
                )
                break
            elif isinstance(result, Correlation):
                logger.info(
                    f"Found correlation for required node '{required_node}' in graph: {result.identifier}"
                )
                break
            else:
                raise ValueError(
                    f"In check_input: found result for '{required_node}': but expected a Correlation or a pint.Quantity, got {result} of type {type(result)}"
                )
    return result


def resolve_correlation(
    corr: Correlation, resolved_quantities: dict, graph: list[object]
) -> pint.Quantity:
    corr_args = inspect.signature(corr.function).parameters.keys()
    for arg in corr_args:
        logger.info(
            f"Resolving argument '{arg}' for correlation '{corr.identifier}'..."
        )
        find_in_graph(arg, resolved_quantities, graph)

    assert all(arg in resolved_quantities for arg in corr_args), (
        f"Could not resolve all arguments for correlation '{corr.identifier}'. "
        f"Missing arguments: {[arg for arg in corr_args if arg not in resolved_quantities]}"
    )

    return corr(**{arg: resolved_quantities[arg] for arg in corr_args})
