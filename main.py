import model
import numpy as np

ANIMATE = True
SHOW_ACTIVITY = False


def main():
    import sys
    import os
    import yaml
    from animation import create_animation
    import pandas as pd

    INPUT_PATH = os.path.join(os.getcwd(), sys.argv[1])
    OUTPUT_PATH = os.path.join(
        os.getcwd(), sys.argv[1] + "_out" if len(sys.argv) < 3 else sys.argv[2]
    )

    def get_input(input_path):
        with open(input_path, "r") as file:
            params = yaml.safe_load(file)
        return params

    def setup_yaml_numpy():
        """Tells PyYAML to represent numpy types in human readable way"""

        def numpy_representer(dumper, data):
            # Convert numpy scalar to a standard Python type
            return dumper.represent_data(data.item())

        # Register the representer for used numpy types (can add other types of needed)
        yaml.add_representer(np.float64, numpy_representer)

    def save_output(results_dict, output_path, input_dict=None, properties_dict=None):
        from datetime import datetime

        setup_yaml_numpy()

        def get_git_hash():
            import subprocess

            try:
                return (
                    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                    .decode("ascii")
                    .strip()
                )
            except:
                return "no-git"

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

    def save_to_csv(array):  # is it a useful feature ?
        # convert the list of arrays into a single 2D matrix
        data_matrix = np.array(array)

        # build the DataFrame
        df = pd.DataFrame(data_matrix)
        df.insert(0, "time", times)  # Put time as the first column

        df.to_csv(OUTPUT_PATH + ".csv", index=False)

    setup_yaml_numpy()

    params = get_input(INPUT_PATH + ".yaml")
    properties = model.compute_properties(params)

    merged_dict = {}
    merged_dict.update(params)
    merged_dict.update(properties)

    t_sparging_hr = [0, 0]  # time interval when sparger is ON
    t_irr_hr = [0, 24]  # time interval when irradiation is ON

    times, c_T_solutions, y_T2_solutions, x_ct, x_y, c_T_volume = model.solve(
        merged_dict,
        t_final=2 * model.days_to_seconds,
        t_irr=[t * model.hours_to_seconds for t in t_irr_hr],
        t_sparging=([t * model.hours_to_seconds for t in t_sparging_hr]),
    )

    # save_to_csv(c_T_volume)

    save_output(
        results_dict=properties, output_path=OUTPUT_PATH + ".yaml", input_dict=params
    )

    if ANIMATE is True:
        # Create interactive animation
        try:
            create_animation(
                times,
                c_T_solutions,
                y_T2_solutions,
                x_ct,
                x_y,
                c_T_volume,
                show_activity=SHOW_ACTIVITY,
            )
        except KeyboardInterrupt:
            exit()


if __name__ == "__main__":
    main()
