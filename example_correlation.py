from dataclasses import dataclass
from sparging.config import ureg, const


@dataclass
class Correlation:
    identifier: str
    function: callable
    source: str | None = None
    description: str | None = None
    input_units: list[str] | None = None

    def __call__(self, **kwargs):

        # check the dimensions are correct
        if self.input_units is not None:
            for arg_name, expected_dimension in zip(kwargs, self.input_units):
                arg = kwargs[arg_name]
                if not isinstance(arg, ureg.Quantity):
                    raise ValueError(
                        f"Invalid input: expected a pint.Quantity with units of {expected_dimension}, got {arg} of type {type(arg)}"
                    )
                if not arg.check(expected_dimension):
                    raise ValueError(
                        f"Invalid input: expected dimensions of {expected_dimension}, got {arg.dimensionality}"
                    )
        return self.function(**kwargs).to_base_units()


h_highbie = Correlation(
    identifier="h_l_higbie",
    function=lambda D_l, u_g, d_b: ((D_l * u_g) / (const.pi * d_b)) ** 0.5,
    source="Higbie 1935",
    description="mass transfer coefficient for tritium in liquid FLiBe using Higbie penetration model",
    input_units=("meter**2/s", "meter/s", "meter"),
)

D_l = 2 * ureg("m**2/s")

print(h_highbie(D_l=D_l, u_g=0.1 * ureg("m/s"), d_b=0.01 * ureg("m")))
# h_highbie(1.0, 2.0, 3.0)
