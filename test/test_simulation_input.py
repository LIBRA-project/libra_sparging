import sparging
from sparging.config import ureg
from sparging import all_correlations
from sparging.inputs import (
    ColumnGeometry,
    BreederMaterial,
    OperatingParameters,
    SpargingParameters,
)

# define standard LIBRA input parameters to be used in multiple tests
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
    irradiation_signal=1,  # ignored for now
    t_sparging=60 * ureg.s,  # TODO implement
    tbr=0.1 * ureg("triton / neutron"),
    n_gen_rate=1e9 * ureg("neutron / s"),
)

sparging_params = SpargingParameters(
    h_l=all_correlations("h_l_briggs"),
)


def test_find_in_graph(tmp_path):
    from sparging.inputs import find_in_graph
    import difflib
    import logging
    from pathlib import Path

    reference_log_path = Path(__file__).with_name("test_find_in_graph.reference.log")
    generated_log_path = Path(tmp_path).joinpath("test_find_in_graph.generated.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        handlers=[logging.FileHandler(generated_log_path, mode="w")],
        force=True,  # reset handlers so pytest/previous tests don't interfere
    )

    find_in_graph("drho", {}, [geom, flibe, operating_params, sparging_params])

    logging.shutdown()

    assert reference_log_path.exists(), (
        f"Reference log not found at {reference_log_path}. "
        f"Create/update it from {generated_log_path} once output is validated."
    )

    generated_text = generated_log_path.read_text(encoding="utf-8")
    reference_text = reference_log_path.read_text(encoding="utf-8")

    if generated_text != reference_text:
        diff = "\n".join(
            difflib.unified_diff(
                reference_text.splitlines(),
                generated_text.splitlines(),
                fromfile=str(reference_log_path.name),
                tofile=str(generated_log_path.name),
                lineterm="",
            )
        )
        raise AssertionError(f"Log output mismatch:\n{diff}")
