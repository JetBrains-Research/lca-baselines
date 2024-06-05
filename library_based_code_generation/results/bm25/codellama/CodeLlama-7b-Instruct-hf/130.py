  import mpi_sppy as msp
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_names", nargs="+", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--solution_dir", required=False)
    return parser.parse_args()

def main():
    args = parse_args()
    config = msp.Config(args.config_file)
    scenario_creator = msp.ScenarioCreator(config)
    scenarios = scenario_creator.create_scenarios(args.scenario_names)
    if len(scenarios) > 10:
        raise ValueError("Too many scenarios")
    hub = msp.WheelSpinner(config)
    for scenario in scenarios:
        hub.add_spoke(scenario)
    if args.solution_dir:
        hub.spin_the_wheel()
        hub.write_solution(args.solution_dir)
    else:
        hub.spin_the_wheel()

if __name__ == "__main__":
    main()