from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime
from sparging import config as helpers
from sparging.closure import Correlation, all_closures
import pint
from typing import List
import inspect
import numpy as np
import logging
from sparging.config import ureg, const_R
from collections.abc import Callable
import networkx as nx


logger = logging.getLogger(__name__)


def _ppl_factor(Pi: float) -> float:
    """Pi / (1 - exp(-Pi)): correction of the SPP extraction time for the partial
    pressure building up in the bubbles. Tends to 1 for Pi << 1 and to Pi for Pi >> 1."""
    return 1.0 if abs(Pi) < 1e-12 else -Pi / np.expm1(-Pi)


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
    ndot_g0: pint.Quantity
    P_top: pint.Quantity
    Vdot_g0: pint.Quantity | None = None
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
    temperature: pint.Quantity
    K_s: pint.Quantity
    rho_l: pint.Quantity
    E_g: pint.Quantity
    E_l: pint.Quantity
    Q_T: pint.Quantity
    signal_irr: Callable[[pint.Quantity], float] = lambda t: 0
    """callable = f:R+ (time) -> [0,1] """
    signal_sparging: Callable[[pint.Quantity], float] = lambda t: 1
    """callable = f:R+ (time) -> [0,1] """
    profile_source_T: Callable[[float], float] | None = None
    """callable = f:[0,1] -> R+, it takes a dimensionless coordinate: (z / height)"""
    c_T2_init: pint.Quantity = 0 * ureg("molT2/m**3")
    profile_c_T2_init: Callable[[float], pint.Quantity] | None = None
    """callable = f:[0,1] -> R+, it takes a dimensionless coordinate: (z / height)"""
    required_scalars = (
        "height",
        "area",
        "temperature",
        "K_s",
        "rho_l",
        "E_g",
        "E_l",
        "Q_T",
    )
    required_profiles = (
        "P_l",
        "P_g",
        "eps_g",
        "a",
        "u_g",
        "h_l",
    )  # these parameters will be used to solve the model
    graph: nx.Graph | None = None
    """ Stores the intermediate parameters and their relationships that built the SimulationInput. 
    It has attributes "nodes" and "edges".
    Each node is accessed by its ID (the param name) and contains a dictionary with properties:
        - value: value of the parameter as pint.Quantity
        - origin: "input" | correlation identifier
    e.g use: mySimulationInput.graph.nodes["height"]["value"]
    """
    # pressure dependant profiles
    P_l: Callable[[pint.Quantity], pint.Quantity] | None = None
    P_g: Callable[[pint.Quantity], pint.Quantity] | None = None
    eps_g: Callable[[pint.Quantity], pint.Quantity] | None = None
    a: Callable[[pint.Quantity], pint.Quantity] | None = None
    u_g: Callable[[pint.Quantity], pint.Quantity] | None = None
    h_l: Callable[[pint.Quantity], pint.Quantity] | None = None

    @property
    def volume(self):
        return self.area * self.height

    @property
    def a_0(self):
        return self.a(0 * ureg.m)

    @property
    def eps_g0(self):
        return self.eps_g(0 * ureg.m)

    @property
    def eps_l0(self):
        return 1 - self.eps_g0

    @property
    def P_l0(self):
        return self.P_l(0 * ureg.m)

    @property
    def P_g0(self):
        return self.P_g(0 * ureg.m)

    @property
    def u_g0(self):
        return self.u_g(0 * ureg.m)

    @property
    def h_l0(self):
        return self.h_l(0 * ureg.m)

    def set_S_T(self, val: pint.Quantity):
        self.Q_T = (val.to("molT/m**3/s") * self.volume).to("molT/s")

    def get_S_T(self) -> pint.Quantity:
        return (self.Q_T / self.volume).to("molT/m**3/s")

    def _height_integral(
        self, profile: Callable[[pint.Quantity], pint.Quantity], unit: str
    ) -> pint.Quantity:
        """integral of a z-dependent profile (evaluated in `unit`) over [0, height]."""
        from scipy.integrate import quad

        def integrand(z_m: float) -> float:
            return profile(z_m * ureg.m).to(unit).magnitude

        height_m = self.height.to("m").magnitude
        return quad(integrand, 0, height_m)[0] * ureg(unit) * ureg.m

    def get_Pi(self, z: pint.Quantity) -> pint.Quantity:
        """Partial pressure number for the hydrodynamic conditions found at height z:
        ratio of the bubble residence time to its saturation time constant.
        If Pi < ~0.1, then gas partial pressure can be neglected in front of liquid concentration, we are in the small partial pressure (SPP) regime,
        If Pi > ~0.1, then partial pressure starts limiting the interfacial mass transfer (PPL regime)
        """
        return (
            self.K_s
            * (const_R * self.temperature)
            * self.height
            * self.a(z)
            * self.h_l(z)
            / self.u_g(z)
        ).to("dimensionless")

    def get_Pi0(self) -> pint.Quantity:
        """Partial pressure number evaluated at the tank bottom (z=0)."""
        return self.get_Pi(0 * ureg.m)

    def get_Pi_ave(self) -> pint.Quantity:
        """Partial pressure number averaged over the tank height. Equals the exponent
        K_s*R*T*int_0^H a*h_l/u_g dz that sets the bubble saturation at the outlet."""
        return (self._height_integral(self.get_Pi, "dimensionless") / self.height).to(
            "dimensionless"
        )

    def get_tau0(self) -> pint.Quantity:
        """extraction time under the small partial pressure (SPP) approximation, with
        the closure relations evaluated at the tank bottom (z=0) instead of averaged
        over the tank height."""
        return (self.eps_l0 / (self.h_l0 * self.a_0)).to("seconds")

    def get_tau_SPP(self) -> pint.Quantity:
        """extraction time under the small partial pressure (SPP) approximation,
        from the eps_l and a*h_l profiles averaged over the tank height."""
        int_eps_l = self._height_integral(lambda z: 1 - self.eps_g(z), "dimensionless")
        int_a_h_l = self._height_integral(lambda z: self.a(z) * self.h_l(z), "1/s")
        return (int_eps_l / int_a_h_l).to("seconds")

    def get_tau_PPL(self) -> pint.Quantity:
        """extraction time in the partial pressure limited (PPL) asymptote Pi >> 1, i.e. the
        limit of get_tau() since Pi/(1-exp(-Pi)) -> Pi. Bubbles leave saturated, so the
        extraction is throttled by the gas throughput rather than by the mass transfer."""
        return (self.get_tau_SPP() * self.get_Pi_ave()).to("seconds")

    def get_tau_PPL_outlet(self) -> pint.Quantity:
        """PPL extraction time written directly from the outlet balance,
        <eps_l>*H*R*T*K_s/u_g(H). Equals get_tau_PPL() when the hydrodynamic parameters are
        uniform; the two differ by the hydrostatic variation alone (a factor ~7 at G_P = 15)."""
        int_eps_l = self._height_integral(lambda z: 1 - self.eps_g(z), "dimensionless")
        return (
            int_eps_l * (const_R * self.temperature) * self.K_s / self.u_g(self.height)
        ).to("seconds")

    def get_tau(self) -> pint.Quantity:
        """extraction time valid in any partial pressure regime: the SPP time scaled
        by the saturation correction Pi/(1-exp(-Pi)), with Pi averaged over the height."""
        return (self.get_tau_SPP() * _ppl_factor(self.get_Pi_ave().magnitude)).to(
            "seconds"
        )

    def get_c_T2_SS(self) -> pint.Quantity:
        return (self.get_S_T() * 1 / (self.h_l0 * self.a_0)).to("molT2/m^3")

    def get_G_mix_pred(self) -> pint.Quantity:
        """Predicted mixing number: ratio of the liquid dispersive mixing time
        scale (H^2/E_l) to the extraction time tau. Governs the "perfectly mixed
        liquid" assumption. "Predicted" since it uses the analytical tau rather
        than a simulated/fitted one."""
        return ((self.height**2 / self.E_l) / self.get_tau()).to("dimensionless")

    def get_G_P(self) -> pint.Quantity:
        """Relative hydrostatic pressure variation along the tank height
        (P_bottom - P_top) / P_top. Governs the "constant hydrodynamic
        parameters along z" assumption."""
        P_top = self.P_l(self.height)
        return ((self.P_l0 - P_top) / P_top).to("dimensionless")

    def get_dP_dx(self) -> pint.Quantity:
        """
        returns 1/c_T * dP_T/dx in the SPP approximation (linearized around P_T = 0))
        """
        return (
            6
            * (const_R * self.temperature)
            * self.h_l0
            / (self.graph.nodes["d_b"]["value"](0 * ureg.m) * self.u_g0)
        ).to("Pa/(mol/m^3)/m")

    def get_Bo(self) -> pint.Quantity:
        """
        returns Bodenstein number = ratio of convective to dispersive transport for the gas phase
        corresponds to Peclet number at the scale of the tank
        """
        v_g0 = self.u_g0 / self.eps_g0
        return (v_g0 * self.height / self.E_g).to("dimensionless")

    def dx_from_Pe(self, Pe: float) -> pint.Quantity:
        """
        returns the spatial step dx corresponding to a given mesh Peclet number
        """
        return (self.E_g * Pe / self.u_g0).to("m")

    def __post_init__(self):
        # make sure there are only pint.Quantity or callables in the input, otherwise raise an error
        for key in self.required_scalars:
            value = getattr(self, key)
            if not isinstance(value, pint.Quantity):
                raise ValueError(
                    f"In {self.__class__.__name__}: Invalid type for '{key}': expected a pint.Quantity, got {value} of type {type(value)}"
                )

    def intermediate_params_dict(self) -> dict:
        """Return {param_name: {value, origin}} from the resolution graph.
        Profiles (callables) are evaluated at z=0 and suffixed '_0'."""
        params = {}
        for node in self.graph.nodes:
            value = self.graph.nodes[node]["value"]
            origin = self.graph.nodes[node]["origin"]
            if callable(value):
                params[f"{node}_0"] = {
                    "value": str(value(0 * ureg.m)),
                    "origin": origin,
                }
            else:
                params[node] = {"value": str(value), "origin": origin}
        return params

    def to_json(self, path: str):
        """Export intermediate parameters for inspection."""
        import json

        output = {
            "metadata": {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
            },
            "intermediate parameters": self.intermediate_params_dict(),
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=4)

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

        for required_key in (*cls.required_scalars, *cls.required_profiles):
            find_in_graph(required_key, discovered_graph, input_objs=input_objects)

        return cls(
            graph=discovered_graph,
            **{
                arg: discovered_graph.nodes[arg]["value"]
                for arg in (*cls.required_scalars, *cls.required_profiles)
            },
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
        if required_node in all_closures:
            result = all_closures(required_node)
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

    assert isinstance(result, pint.Quantity) or callable(result), (
        f"Result for required node '{required_node}' is not a pint.Quantity or callable after resolution, got {result} of type {type(result)}"
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
) -> pint.Quantity | callable:
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
