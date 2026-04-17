import sparging
from sparging.config import ureg
from sparging import all_correlations, CorrelationGroup, Correlation
from sparging.inputs import (
    ColumnGeometry,
    BreederMaterial,
    OperatingParameters,
    SpargingParameters,
    find_in_graph,
    check_input,
    SimulationInput,
)

import pytest
import dataclasses
import difflib
import logging
from pathlib import Path

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
    tbr=0.1 * ureg("triton / neutron"),
    n_gen_rate=1e9 * ureg("neutron / s"),
)

sparging_params = SpargingParameters(
    h_l=all_correlations("h_l_briggs"),
)


def test_from_parameters_success(tmp_path):
    """
    Test that SimulationInput.from_parameters successfully creates a SimulationInput object from minimal input objects
    and that the generated SimulationInput is consistent with the standard input it should yield
    """
    sim_input = SimulationInput.from_parameters(
        geom, flibe, operating_params, sparging_params
    )

    assert isinstance(sim_input, SimulationInput), (
        "Expected from_parameters to return a SimulationInput instance"
    )
    # Check that all fields are populated and have the correct types
    for field in dataclasses.fields(sim_input):
        value = getattr(sim_input, field.name)
        assert isinstance(value, ureg.Quantity), (
            f"Expected field '{field.name}' to be a pint.Quantity, got {type(value)}"
        )

    reference_path = Path(__file__).with_name("standard_input.json")
    generated_path = Path(tmp_path).joinpath("generated_input.json")

    sim_input.to_json(generated_path)

    generated_text = generated_path.read_text(encoding="utf-8")
    reference_text = reference_path.read_text(encoding="utf-8")

    diff = "\n".join(
        difflib.unified_diff(
            reference_text.splitlines(),
            generated_text.splitlines(),
            fromfile=str(reference_path.name),
            tofile=str(generated_path.name),
            lineterm="",
        )
    )
    assert generated_text == reference_text, f"Log output mismatch:\n{diff}"


def test_find_in_graph_logging(tmp_path):
    """
    Test that the `find_in_graph` function logs the expected output when searching for a parameter in the graph.
    """
    # BUILD
    reference_log_path = Path(__file__).with_name("test_find_in_graph.reference.log")
    generated_log_path = Path(tmp_path).joinpath("test_find_in_graph.generated.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        handlers=[logging.FileHandler(generated_log_path, mode="w")],
        force=True,  # reset handlers so pytest/previous tests don't interfere
    )

    # RUN
    find_in_graph("drho", {}, [geom, flibe, operating_params, sparging_params])

    # TEST
    logging.shutdown()

    assert reference_log_path.exists(), (
        f"Reference log not found at {reference_log_path}. "
        f"Create/update it from {generated_log_path} once output is validated."
    )

    generated_text = generated_log_path.read_text(encoding="utf-8")
    reference_text = reference_log_path.read_text(encoding="utf-8")

    diff = "\n".join(
        difflib.unified_diff(
            reference_text.splitlines(),
            generated_text.splitlines(),
            fromfile=str(reference_log_path.name),
            tofile=str(generated_log_path.name),
            lineterm="",
        )
    )
    assert generated_text == reference_text, f"Log output mismatch:\n{diff}"


@pytest.mark.parametrize("in_discovered", (True, False))
def test_find_in_graph_result(in_discovered: bool):
    """
    Test finding a node in the graph.
    This test checks that the `find_in_graph` function can successfully find the `d_b` parameter
    using the provided `ColumnGeometry` and `OperatingParameters`. It also tests both cases where
    `flow_g_vol` is provided in the discovered nodes and where it is not, ensuring that the
    function can handle both scenarios correctly.
    """
    # BUILD
    discovered_nodes = {}
    if in_discovered:
        discovered_nodes["flow_g_vol"] = 0.01 * ureg.m**3 / ureg.s

    # RUN
    find_in_graph(
        "d_b",
        discovered_nodes=discovered_nodes,
        graph=[geom, operating_params],
    )

    # TEST
    assert "d_b" in discovered_nodes, "Expected to find d_b in graph"

    correlation = sparging.all_correlations("d_b")

    flow_g_vol = discovered_nodes.get("flow_g_vol")
    expected_value = correlation(
        flow_g_vol=flow_g_vol,
        nozzle_diameter=geom.nozzle_diameter,
        nb_nozzle=geom.nb_nozzle,
    )

    assert discovered_nodes["d_b"] == expected_value, (
        f"Expected d_b to be {expected_value}, got {discovered_nodes['d_b']}"
    )


@pytest.mark.parametrize("missing_param", ("nb_nozzle", "flow_g_mol", "n_gen_rate"))
def test_find_in_graph_unresolvable(missing_param: str):
    """
    Test that find_in_graph raises an error when a parameter cannot be resolved.
    """
    # BUILD
    broken_geom = dataclasses.replace(geom)
    broken_op_params = dataclasses.replace(operating_params)
    to_find = str()
    match missing_param:
        case "nb_nozzle":
            to_find = "d_b"
            setattr(broken_geom, missing_param, None)
        case "flow_g_mol":
            to_find = "d_b"
            setattr(broken_op_params, missing_param, None)
        case "n_gen_rate":
            to_find = "source_T"
            setattr(broken_op_params, missing_param, None)
    # RUN
    with pytest.raises(
        ValueError,
        match=f"Could not find path to required node '{missing_param}' in the graph or in the default correlations",
    ):
        find_in_graph(
            to_find,
            discovered_nodes={},  # no discovered nodes provided
            graph=[
                broken_geom,
                broken_op_params,
            ],  # missing necessary parameters for d_b correlation
        )


@pytest.mark.parametrize("required_node", ("flow_g_mol", "non_existent_param"))
def test_check_input_none(required_node: str):
    """
    Test that check_input returns None when the required node is not found in the graph.
    """
    # BUILD
    broken_op_params = dataclasses.replace(operating_params, flow_g_mol=None)
    # RUN
    result = check_input(required_node, [geom, broken_op_params])
    # TEST
    assert result is None, f"Expected None for non-existent parameter, got {result}"


def test_check_input_unexpected_type():
    """
    Test that check_input raises an error when it finds a parameter in the graph but it is not a pint.Quantity or Correlation.
    """
    # BUILD
    broken_op_params = dataclasses.replace(operating_params, tbr=13)
    # RUN
    result = -1
    with pytest.raises(
        ValueError,
        match="In check_input: found result for 'tbr': but expected a Correlation or a pint.Quantity, got 13 of type <class 'int'>",
    ):
        result = check_input("tbr", [geom, broken_op_params])
        assert False, f"Expected ValueError, got {result}"


def test_correlation_group():
    """
    Test that CorrelationGroup __contains__ and __call__ behave as expected.
    """
    # BUILD
    corr_str = "dummy_corr"
    corr_str_typo = "dummy_coor"
    corr_str_wrong_type = ureg.Quantity(5, ureg.s)

    my_corr_obj = Correlation(
        identifier=corr_str,
        function=lambda: None,
        corr_type="dummy",
        input_units=[],
    )
    corr_group = CorrelationGroup([my_corr_obj])

    # TEST __contains__
    if corr_str in corr_group:  # should be found
        pass
    if my_corr_obj in corr_group:  # should be found
        pass
    if corr_str_typo in corr_group:  # should not be found
        assert False, (
            f"Expected '{corr_str_typo}' not to be found in CorrelationGroup, but it was"
        )
    with pytest.raises(
        TypeError,
        match="Type not valid for correlation group membership check, expected str or Correlation, got <class 'pint.Quantity'>",
    ):
        if corr_str_wrong_type in corr_group:  # wrong type, should raise TypeError
            pass

    # TEST __call__
    correlation_found = corr_group(corr_str)  # return a Correlation
    assert isinstance(correlation_found, Correlation), (
        "Expected to find dummy_corr Correlation"
    )

    with pytest.raises(
        ValueError,
        match=f"Correlation with identifier {corr_str_typo} not found",
    ):
        corr_group(corr_str_typo)  # non-existent correlation, should raise ValueError


def test_correlation():
    """
    Test Correlation raises an error when:
    - providing wrong input (missing arguments/wrong dimensionality)"""

    # BUILD
    my_corr_obj = Correlation(
        identifier="dummy_corr",
        function=lambda a, b: a * ureg.s,
        corr_type="dummy",
        input_units=[
            "m"
        ],  # TODO should raise an error if input_units length doesn't match function arguments when constructing the correlation
    )

    # RUN
    assert isinstance(my_corr_obj(a=5 * ureg.m, b=None), ureg.Quantity), (
        "Expected a pint.Quantity as output"
    )  # should work

    with pytest.raises(ValueError, match="expected a pint.Quantity"):
        my_corr_obj(a=5, b=None)  # wrong input type, should be a pint.Quantity
    with pytest.raises(ValueError, match="expected dimensions"):
        my_corr_obj(a=5 * ureg.s, b=None)  # wrong units, should raise ValueError
