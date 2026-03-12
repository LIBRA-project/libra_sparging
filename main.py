import model

def main():
    import sys
    import os
    import yaml
    import json

    INPUT_PATH = os.path.join(os.getcwd(), sys.argv[1])
    OUTPUT_PATH = os.path.join(os.getcwd(), "output.json")
    with open(INPUT_PATH, 'r') as file:
        params = yaml.safe_load(file)
    print("params:", params)

    properties = model.compute_properties(params)
    print("properties:", properties)

    with open(OUTPUT_PATH, "w") as f:
        json.dump([params,properties], f, indent=4)

if __name__ == "__main__":
    main()  