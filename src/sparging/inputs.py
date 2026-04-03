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


@dataclass
class BreederMaterial:
    name: str
    diffusivity: pint.Quantity | Correlation
    solubility: pint.Quantity | Correlation
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


@dataclass
class SpargingParameters:
    h_l: pint.Quantity | Correlation
    eps_g: pint.Quantity | Correlation
    u_g0: pint.Quantity | Correlation
    d_b: pint.Quantity | Correlation
    rho_g: pint.Quantity | Correlation


@dataclass
class SimulationInput:
    height: pint.Quantity
    area: pint.Quantity
    u_g0: pint.Quantity | callable
    temperature: pint.Quantity
    h_l: pint.Quantity | callable
    K_s: pint.Quantity | callable
    P_0: pint.Quantity
    eps_g: pint.Quantity | callable
    eps_l: pint.Quantity | callable
    E_g: pint.Quantity | callable
    D_l: pint.Quantity | callable
    source_T: pint.Quantity | callable

    def __post_init__(self):
        # make sure there are only pint.Quantity or callables in the input, otherwise raise an error
        for key in self.__dataclass_fields__.keys():
            value = getattr(self, key)
            if not isinstance(value, pint.Quantity):
                raise ValueError(
                    f"Invalid input for '{key}': expected a pint.Quantity, got {value} of type {type(value)}"
                )

        self._name_to_method = {"Re": self.get_Re, "Eo": all_correlations("Eo")}

    def resolve(self) -> "SimulationInput":
        arguments_as_quantities = {
            key: getattr(self, key)
            for key in self.__dict__.keys()
            if isinstance(getattr(self, key), pint.Quantity)
            # TODO handle case when no units are given (float) to attach default units
        }

        for key in self.__dict__.keys():
            value = getattr(self, key)
            # if it's a correlation
            if callable(value):
                self.resolve_parameters(key, value, arguments_as_quantities)

        new_input = SimulationInput(
            **{
                key: value
                for key, value in arguments_as_quantities.items()
                if key in self.__dataclass_fields__.keys()
            }
        )
        return new_input

    @classmethod
    def from_parameters(
        cls,
        column_geometry: ColumnGeometry,
        breeder_material: BreederMaterial,
        operating_params: OperatingParameters,
        sparging_params: SpargingParameters,
    ):
        all_params = {
            **column_geometry.__dict__,
            **breeder_material.__dict__,
            **operating_params.__dict__,
            **sparging_params.__dict__,
        }
        # remove None from all_params
        all_params = {
            key: value for key, value in all_params.items() if value is not None
        }
        arguments_as_pint_quantities = {}
        for key, value in all_params.items():
            if isinstance(value, pint.Quantity):
                arguments_as_pint_quantities[key] = value

        required_keys = cls.__dataclass_fields__.keys()
        for key in required_keys:
            if key in all_params.keys():
                value = all_params[key]
                if isinstance(value, Correlation):
                    resolve_parameters(
                        key, value.function, arguments_as_pint_quantities, all_params
                    )
            else:
                raise ValueError(f"Missing required parameter '{key}'")

        return cls(**arguments_as_pint_quantities)


def resolve_parameters(key: str, value: callable, args_quant: dict, all_args: dict):
    corr_args = inspect.signature(value).parameters.keys()
    if all(arg in args_quant for arg in corr_args):
        args_quant[key] = value(**{arg: args_quant[arg] for arg in corr_args})

    else:
        for arg in corr_args:
            if arg not in args_quant.keys():
                # means it's a callable/correlation
                if arg in all_args.keys():
                    arg_value = all_args[arg]
                else:
                    arg_value = all_correlations(arg)
                    all_args[arg] = arg_value  # cache it for future use

                    assert isinstance(arg_value, Correlation), (
                        f"Expected a correlation for argument '{arg}', got {arg_value} of type {type(arg_value)}"
                    )
                if callable(arg_value):
                    resolve_parameters(arg, arg_value.function, args_quant, all_args)

        # after resolving all arguments, we can resolve the correlation itself
        try:
            args_quant[key] = value(**{arg: args_quant[arg] for arg in corr_args})
        except Exception as e:
            breakpoint()
