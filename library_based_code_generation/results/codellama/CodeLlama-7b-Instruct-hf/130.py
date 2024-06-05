  import argparse
import mpi_sppy as mpi
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", type=str, help="List of scenario names")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--solution_dir", type=str, help="Directory to write solution to")
    return parser.parse_args()

def setup_config(config_file):
    config = mpi.Config()
    config.read(config_file)
    return config

def setup_scenarios(scenario_names):
    scenarios = []
    for scenario_name in scenario_names:
        scenario = mpi.Scenario(scenario_name)
        scenarios.append(scenario)
    return scenarios

def setup_extensions(config, scenarios):
    extensions = []
    for scenario in scenarios:
        extension = mpi.Extension(config, scenario)
        extensions.append(extension)
    return extensions

def setup_spokes(config, scenarios):
    spokes = []
    for scenario in scenarios:
        spoke = mpi.Spoke(config, scenario)
        spokes.append(spoke)
    return spokes

def create_wheel_spinner(hub, spokes):
    wheel_spinner = mpi.WheelSpinner(hub, spokes)
    return wheel_spinner

def spin_wheel(wheel_spinner):
    wheel_spinner.spin()

def write_solution(solution, solution_dir):
    if solution_dir is not None:
        solution_file = os.path.join(solution_dir, "solution.txt")
        with open(solution_file, "w") as f:
            f.write(solution)

def main():
    args = parse_args()
    config = setup_config(args.config)
    scenarios = setup_scenarios(args.scenarios)
    extensions = setup_extensions(config, scenarios)
    spokes = setup_spokes(config, scenarios)
    hub = mpi.Hub(config, scenarios, extensions, spokes)
    wheel_spinner = create_wheel_spinner(hub, spokes)
    spin_wheel(wheel_spinner)
    solution = wheel_spinner.get_solution()
    write_solution(solution, args.solution_dir)

if __name__ == "__main__":
    main()