import src.sparging.model as model
import sys
import os
from sparging.animation import create_animation
from sparging.helpers import get_input


ANIMATE = True
SHOW_ACTIVITY = True
YAML_INPUT_PATH = os.path.join(os.getcwd(), sys.argv[1])
OUTPUT_FOLDER = os.path.join(
    os.getcwd(), sys.argv[1].split(".")[0].replace("_input", "")
)

if __name__ == "__main__":
    params = get_input(YAML_INPUT_PATH)
    sim_input = model.SimulationInput(params)

    # TODO integrate to input file
    t_sparging_hr = [24, 1e20]  # time interval when sparger is ON
    t_irr_hr = [0, 96]  # time interval when irradiation is ON
    t_final = 6 * model.days_to_seconds

    results = model.solve(
        sim_input,
        t_final=t_final,
        t_irr=[t * model.hours_to_seconds for t in t_irr_hr],
        t_sparging=[t * model.hours_to_seconds for t in t_sparging_hr],
    )

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    results.to_yaml(os.path.join(OUTPUT_FOLDER, "restart.yaml"))
    results.to_json(os.path.join(OUTPUT_FOLDER, "output.json"))
    # results.profiles_to_csv(os.path.join(OUTPUT_FOLDER, "profiles"))

    if ANIMATE is True:
        # Create interactive animation
        try:
            create_animation(results, show_activity=SHOW_ACTIVITY)
        except KeyboardInterrupt:
            pass
