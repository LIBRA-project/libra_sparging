import model
import sys
import os
import yaml
from animation import create_animation
from helpers import get_input, setup_yaml_numpy, get_git_hash

ANIMATE = True
SHOW_ACTIVITY = True
INPUT_PATH = os.path.join(os.getcwd(), sys.argv[1])
OUTPUT_PATH = os.path.join(
    os.getcwd(), sys.argv[1] + "_out" if len(sys.argv) < 3 else sys.argv[2]
)


def save_output(results_dict, output_path, input_dict=None, properties_dict=None):
    from datetime import datetime

    setup_yaml_numpy()

    # structure the output
    output = {
        "metadata": {
            "git_commit": get_git_hash(),
            "date": datetime.now().isoformat(),
        },
    }
    if input_dict is not None:
        output["input parameters"] = input_dict
    if properties_dict is not None:
        output["calculated properties"] = properties_dict
    output["results"] = results_dict

    with open(output_path, "w") as f:
        yaml.dump(output, f, sort_keys=False)


if __name__ == "__main__":
    setup_yaml_numpy()

    params = get_input(INPUT_PATH + ".yaml")
    properties = model.compute_properties(params)

    # breakpoint()
    merged_dict = {}
    merged_dict.update(params)
    merged_dict.update(properties)

    t_sparging_hr = [0, 1e20]  # time interval when sparger is ON
    t_irr_hr = [0, 96]  # time interval when irradiation is ON
    t_final = 10 * model.days_to_seconds

    results = model.solve(
        merged_dict,
        t_final=t_final,
        t_irr=[t * model.hours_to_seconds for t in t_irr_hr],
        t_sparging=[t * model.hours_to_seconds for t in t_sparging_hr],
    )
    # save_to_csv(c_T_volume)

    save_output(
        results_dict=properties, output_path=OUTPUT_PATH + ".yaml", input_dict=params
    )

    # breakpoint()
    if ANIMATE is True:
        # Create interactive animation
        try:
            create_animation(
                results.times,
                results.c_T2_solutions,
                results.y_T2_solutions,
                results.x_ct,
                results.x_y,
                results.inventories_T2_salt,
                source_T2=results.source_T2,
                fluxes_T2=results.fluxes_T2,
                show_activity=SHOW_ACTIVITY,
            )
        except KeyboardInterrupt:
            exit()
