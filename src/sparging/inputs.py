from dataclasses import dataclass
from sparging.correlations import Correlation
import pint

import inspect
import warnings
import numpy as np
from .config import ureg, const_R, const_g, VERBOSE
from .correlations import all_correlations, U_G0_DEFAULT


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
    flow_g_vol: pint.Quantity
    irradiation_signal: pint.Quantity
    t_sparging: pint.Quantity
    P_bottom: pint.Quantity | Correlation | None = None
    P_top: pint.Quantity | None = None
    source_T: pint.Quantity | Correlation | None = (
        None  # source term for tritium generation, in molT/m^3/s
    )


@dataclass
class SpargingParameters:
    h_l: pint.Quantity | Correlation
    eps_g: pint.Quantity | Correlation
    u_g0: pint.Quantity | Correlation
    d_b: pint.Quantity | Correlation
    rho_g: pint.Quantity | Correlation
    E_g: pint.Quantity | Correlation | None = None
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
    eps_g: pint.Quantity
    E_g: pint.Quantity
    D_l: pint.Quantity
    source_T: pint.Quantity

    @property
    def volume(self):
        return self.area * self.height

    def __post_init__(self):
        # make sure there are only pint.Quantity or callables in the input, otherwise raise an error
        for key in self.__dataclass_fields__.keys():
            value = getattr(self, key)
            if not isinstance(value, pint.Quantity):
                raise ValueError(
                    f"Invalid input for '{key}': expected a pint.Quantity, got {value} of type {type(value)}"
                )

    @classmethod
    def from_parameters(
        cls,
        column_geometry: ColumnGeometry,
        breeder_material: BreederMaterial,
        operating_params: OperatingParameters,
        sparging_params: SpargingParameters,
    ):
        # remove None from all_params
        previously_resolved = {}
        all_params = [
            column_geometry,
            breeder_material,
            operating_params,
            sparging_params,
        ]
        needed_values = {}
        required_keys = cls.__dataclass_fields__.keys()
        for needed_key in required_keys:
            for prms in all_params:
                if hasattr(prms, needed_key):
                    value = getattr(prms, needed_key)

                    if isinstance(value, pint.Quantity):
                        needed_values[needed_key] = value  # TODO do we even need this?
                    elif isinstance(value, Correlation):
                        quantity = resolve_correlation(
                            value,
                            column_geometry,
                            breeder_material,
                            operating_params,
                            sparging_params,
                            previously_resolved=previously_resolved,
                        )
                        needed_values[needed_key] = quantity
                    elif value is None:
                        # try to find a default correlation for this key
                        print(
                            f"Value for '{needed_key}' is None, looking for a default correlation..."
                        )
                        if default_correlation := all_correlations(needed_key):
                            quantity = resolve_correlation(
                                default_correlation,
                                column_geometry,
                                breeder_material,
                                operating_params,
                                sparging_params,
                                previously_resolved=previously_resolved,
                            )
                            needed_values[needed_key] = quantity

        return cls(**needed_values)


def resolve_correlation(
    correlation: Correlation,
    column_geometry,
    breeder_material,
    operating_params,
    sparging_params,
    previously_resolved={},
):
    all_params = [
        column_geometry,
        breeder_material,
        operating_params,
        sparging_params,
    ]
    corr_args = inspect.signature(correlation.function).parameters.keys()
    for arg in corr_args:
        for prms in all_params:
            if hasattr(prms, arg):
                value = getattr(prms, arg)
                if isinstance(value, pint.Quantity):
                    previously_resolved[arg] = value  # TODO do we even need this?
                    break
                elif isinstance(value, Correlation):
                    previously_resolved[arg] = resolve_correlation(
                        value,
                        column_geometry,
                        breeder_material,
                        operating_params,
                        sparging_params,
                        previously_resolved,
                    )
                    break

        # if the arg is not in the params, find a default correlation
        if arg not in previously_resolved.keys():
            print(
                f"Argument '{arg}' not found in input parameters, looking for a default correlation..."
            )
            if default_correlation := all_correlations(arg):
                previously_resolved[arg] = resolve_correlation(
                    default_correlation,
                    column_geometry,
                    breeder_material,
                    operating_params,
                    sparging_params,
                    previously_resolved,
                )

    assert all(arg in previously_resolved for arg in corr_args), (
        f"Could not resolve all arguments for correlation '{correlation.identifier}'. "
        f"Missing arguments: {[arg for arg in corr_args if arg not in previously_resolved]}"
    )
    return correlation.function(**{arg: previously_resolved[arg] for arg in corr_args})
