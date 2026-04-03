import sparging


def test_get():
    # BUILD
    input_dict = {"h_l": "from h_l_malara"}
    # RUN
    quantity = sparging.model.get_quantity_or_correlation(input_dict, "h_l")

    # TEST
    assert callable(quantity), f"Expected a correlation function, got {quantity}"


# def test_get_source_T():
#     # BUILD
#     params = sparging.helpers.get_input("test/test_input.yml")
#     # RUN
#     quantity = sparging.model.get_quantity_or_correlation(params, "source_T")

#     # TEST
#     assert callable(quantity), "Expected a correlation function"


def test_get_from_file():
    # BUILD
    params = sparging.helpers.get_input("test/test_input.yml")
    sparging.model.SimulationInput(params)
