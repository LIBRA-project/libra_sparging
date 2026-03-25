import model
import sys
import os
from animation import create_animation
from helpers import get_input

from pint import UnitRegistry

ureg = UnitRegistry()
ureg.formatter.default_format = "~P"

ANIMATE = True
SHOW_ACTIVITY = True
INPUT_PATH = os.path.join(os.getcwd(), sys.argv[1])
OUTPUT_PATH = os.path.join(
    os.getcwd(), sys.argv[1] + "_out" if len(sys.argv) < 3 else sys.argv[2]
)


if __name__ == "__main__":
    params = get_input(INPUT_PATH + ".yaml")
    properties = model.compute_properties(params)

    # breakpoint()
    merged_dict = {}
    merged_dict.update(params)
    merged_dict.update(properties)

    t_sparging_hr = [24, 1e20]  # time interval when sparger is ON
    t_irr_hr = [0, 96]  # time interval when irradiation is ON
    t_final = 6 * model.days_to_seconds

    results = model.solve(
        merged_dict,
        t_final=t_final,
        t_irr=[t * model.hours_to_seconds for t in t_irr_hr],
        t_sparging=[t * model.hours_to_seconds for t in t_sparging_hr],
    )
    # save_to_csv(c_T_volume)

    results.to_yaml(OUTPUT_PATH + ".yaml", inputs=params, properties=properties)
    results.to_json(OUTPUT_PATH + ".json", inputs=params, properties=properties)
    results.profiles_to_csv(OUTPUT_PATH + "_profiles")
    # breakpoint()
    if ANIMATE is True:
        # Create interactive animation
        try:
            create_animation(results, show_activity=SHOW_ACTIVITY)
        except KeyboardInterrupt:
            pass
