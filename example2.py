from sparging.config import ureg, const_R, const_g
from sparging import all_correlations
from sparging.model import solve
from sparging.inputs import (
    ColumnGeometry,
    BreederMaterial,
    OperatingParameters,
    SpargingParameters,
    SimulationInput,
)
import numpy as np


def source_from_tbr(tbr, n_gen_rate, tank_volume):
    return tbr * n_gen_rate / tank_volume


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
    # flow_g_vol=0.1 * ureg.m**3 / ureg.s,
    flow_g_mol=400 * ureg.sccm,
    irradiation_signal=1,  # ignored for now
    t_sparging=60 * ureg.s,
    tbr=0.1 * ureg("triton / neutron"),
    n_gen_rate=1e9 * ureg("neutron / s"),
)

sparging_params = SpargingParameters(
    h_l=all_correlations("h_l_malara"),
)

# class method from_parameters that takes in objects like ColumnGeometry, BreederMaterial, OperatingParameters and returns a SimulationInput object with the appropriate correlations for the given parameters. This method should be able to handle cases where some of the parameters are already provided as correlations and should not overwrite them.
my_input = SimulationInput.from_parameters(
    geom, flibe, operating_params, sparging_params
)

assert isinstance(my_input, SimulationInput)

# inputs = SimulationInput(
#     nozzle_diameter=0.001 * ureg.m,
#     nb_nozzle=12 * ureg.dimensionless,
#     D_l=lambda T: ureg.Quantity(
#         9.3e-7
#         * ureg.m**2
#         / ureg.s
#         * np.exp(-42e3 * ureg("J/mol") / (const_R * T.to("kelvin")))
#     ),
#     K_s=1e-5 * ureg.mol / ureg.m**3 / ureg.Pa,
#     tank_height=tank_height * ureg.m,
#     tank_area=0.2 * ureg.m**2,  # cross-sectional area of the tank [m^2]
#     u_g0=corr.new_get_u_g0,  # superficial gas velocity [m/s]
#     T=600 * ureg.celsius,  # temperature
#     h_l=corr.get_h_malara,  # mass transfer coefficient [m/s]
#     P_0=1 * ureg.atm,  # pressure [Pa]
#     eps_g=0.01 * ureg.dimensionless,
#     eps_l=0.99 * ureg.dimensionless,  # liquid void fraction
#     E_g=0.0 * ureg.kg / ureg.m**3 / ureg.s,
#     source_T=source_from_tbr,
#     extra_args={
#         "d_b": corr.get_d_b,
#         "tbr": 0.1 * ureg("triton / neutron"),
#         "n_gen_rate": 1e9 * ureg("neutron / s"),
#     },
# )

# inputs = get_simulation_input(
#     geometry=ColumnGeometry(diameter=0.5 * ureg.m, height=tank_height * ureg.m),
#     breeder=BreederMaterial(...),
#     operating_params=OperatingParameters(...),
# )

# inputs.nb_nozzle = 20000

# inputs = SimulationInput(
#     nozzle_diameter=0.01 * ureg.m,  # diameter of the gas injection nozzle [m]
#     nb_nozzle=12,
#     tank_height=tank_height * ureg.m,  # height of the liquid in the tank [m]
#     tank_area=1.0 * ureg.m**2,  # cross-sectional area of the tank [m^2]
#     u_g0=0.1,  # superficial gas velocity [m/s]
#     T=300 * ureg.K,  # temperature [K]
#     h_l=corr.get_h_malara,  # mass transfer coefficient [m/s] (can be a number or a correlation function)
#     K_S=2,
#     P_0=1 * ureg.atm,  # pressure [Pa]
#     eps_g=0.01,  # gas void fraction (can be a number or a correlation function)
#     eps_l=0.99,  # liquid void fraction (can be a number or a correlation function
#     E_g=0.0,  # gas phase source term [kg/m^3/s] (can be a number or a correlation function
#     D_l=3,  # diffusivity of tritium in liquid FLiBe [m^2/s]
#     source_T=30
#     * ureg.molT
#     / ureg.m**3
#     / ureg.s,  # tritium source term [kg/m^3/s] (can be a number or a correlation function)
# )

# inputs.to_yaml(f"input_{tank_height}m.yml")
# inputs.to_json(f"input_{tank_height}m.json")

# unpacked_inputs = inputs.resolve()

# inputs = SimulationInput.from_yaml(f"input_{tank_height}m.yml")
# inputs.T = 500
# inputs.to_yaml(f"input_{tank_height}m_modified.yml")
# output = solve(unpacked_inputs)

# save output to file
# output.profiles_to_csv(f"output_{tank_height}m.csv")

# plot results
# from sparging import plotting
# plotting.plot_animation(output)

print(my_input)
output = solve(
    my_input,
    t_final=6 * ureg.days,
    t_irr=[0, 96] * ureg.h,
    t_sparging=[24, 1e9] * ureg.h,
)
from sparging import animation

animation.create_animation(output, show_activity=True)
