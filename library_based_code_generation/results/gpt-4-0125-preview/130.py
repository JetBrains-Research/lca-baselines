```python
import argparse
import mpi4py.MPI as mpi
from mpisppy.utils import baseparsers, vanilla
from mpisppy.utils.sputils import create_EF
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.cylinders.spoke import Spoke

def parse_args():
    parser = argparse.ArgumentParser(description="Set up configuration for scenario-based problem.")
    baseparsers.add_parser_args(parser)
    parser.add_argument("--num-scenarios", help="Number of scenarios", type=int, default=10)
    parser.add_argument("--solution-directory", help="Directory to write solution files", type=str, default=None)
    args = parser.parse_args()
    config = {
        "num_scenarios": args.num_scenarios,
        "solution_directory": args.solution_directory,
    }
    return config

def main():
    config = parse_args()
    num_scenarios = config["num_scenarios"]
    solution_directory = config["solution_directory"]

    # Example scenario creator and names
    def scenario_creator(scenario_name):
        # Placeholder for actual scenario creation logic
        pass

    scenario_names = [f"Scenario{i+1}" for i in range(num_scenarios)]

    # Check if the number of scenarios is valid
    available_scenarios = [10, 20, 30]  # Example list of available scenarios
    if num_scenarios not in available_scenarios:
        raise ValueError(f"Number of scenarios {num_scenarios} is not in the list of available scenarios.")

    # Set up extensions and spokes based on configuration
    # This is a placeholder for actual extension and spoke setup
    extensions = None
    spokes = []

    # Example of setting up a hub and spokes
    hub_dict = {
        "hub_class": None,  # Placeholder for actual hub class
        "hub_kwargs": {"options": {"num_scenarios": num_scenarios}},
        "opt_kwargs": {"options": {"solver_name": "gurobi"}},
    }

    # Placeholder for spoke setup
    # spoke_dict = {
    #     "spoke_class": SpokeClass,
    #     "spoke_kwargs": {"options": {"some_option": value}},
    # }

    # Create WheelSpinner object and spin the wheel
    wheel = WheelSpinner(hub_dict, spokes)
    wheel.spin()

    # Write solution to file if solution directory is provided
    if solution_directory:
        with open(f"{solution_directory}/solution.txt", "w") as f:
            f.write("Solution placeholder")  # Placeholder for actual solution writing logic

if __name__ == "__main__":
    main()
```