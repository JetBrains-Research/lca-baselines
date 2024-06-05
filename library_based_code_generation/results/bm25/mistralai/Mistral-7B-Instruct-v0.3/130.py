 Here is the Python code based on your instructions:

```python
import argparse
import sys
from mpi4py import MPI
from mpi_sppy.scenario_creator import ScenarioCreator
from mpi_sppy.extensions import ProductionCostFunction, FWPH, PH, APH
from mpi_sppy.rules import calculate_spinning_reserve_available_rule_part_a, enforce_storage_input_limits_rule_part_a, enforce_storage_output_limits_rule_part_a
from mpi_sppy.wheels import WheelSpinner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, nargs='+', required=True)
    parser.add_argument('--solution_dir', type=str)
    args = parser.parse_args()
    return args

def setup_config(args):
    # Set up configuration object with various parameters
    config = {
        'scenario': args.scenario,
        'solution_dir': args.solution_dir
    }
    return config

def main(config):
    # Set up scenario creator and list of scenario names
    creator = ScenarioCreator()
    scenarios = config['scenario']

    # Check if the number of scenarios is in a predefined list of available scenarios
    if not set(scenarios).issubset(creator.get_available_scenarios()):
        raise ValueError("Invalid scenario(s) provided.")

    # Set up different extensions and spokes for the scenario
    extensions = {
        'production_cost': ProductionCostFunction(),
        'fwph': FWPH(),
        'ph': PH(),
        'aph': APH()
    }

    # Create a WheelSpinner object with the hub and spokes
    wheel = WheelSpinner(creator, extensions)

    # Spin the wheel and write the solution to a file if a solution directory is provided
    if config['solution_dir']:
        wheel.spin_the_wheel(config['solution_dir'])
        if wheel.first_stage_solution:
            wheel.write_spin_the_wheel_first_stage_solution()
        if wheel.tree_solution:
            wheel.write_spin_the_wheel_tree_solution()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    config = setup_config(args)
    main(config)
```

This code imports necessary modules and functions from the mpi-sppy library and other necessary libraries. It defines a function to parse arguments from the command line and set up a configuration object with various parameters. Then, it defines a main function that uses the parsed arguments to set up a scenario creator and a list of scenario names. The main function also checks if the number of scenarios is in a predefined list of available scenarios. Depending on the configuration, the main function sets up different extensions and spokes for the scenario. The main function also creates a WheelSpinner object with the hub and spokes, spins the wheel, and writes the solution to a file if a solution directory is provided. Finally, the code calls the main function if the script is run as the main program.