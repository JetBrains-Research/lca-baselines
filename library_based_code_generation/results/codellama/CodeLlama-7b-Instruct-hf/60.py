  import os
import sys
import subprocess
from basilisk.controller import MonteCarloController
from basilisk.retention import RetentionPolicy

def rerun_monte_carlo(scenario_name, num_processes, run_numbers, retention_policy=None):
    # Set up the Monte Carlo controller
    controller = MonteCarloController(scenario_name)

    # Set the initial conditions directory
    controller.set_initial_conditions_directory("initial_conditions")

    # Set the archive directory
    controller.set_archive_directory("archive")

    # Set the execution count
    controller.set_execution_count(run_numbers)

    # Set whether to disperse seeds or archive parameters
    controller.set_disperse_seeds(False)
    controller.set_archive_parameters(True)

    # Add the specified retention policy
    if retention_policy:
        controller.add_retention_policy(retention_policy)

    # Run the initial conditions
    controller.run_initial_conditions()

if __name__ == "__main__":
    # Get the scenario name, number of processes, and run numbers from the command line arguments
    scenario_name = sys.argv[1]
    num_processes = int(sys.argv[2])
    run_numbers = int(sys.argv[3])

    # Create a retention policy
    retention_policy = RetentionPolicy(
        "retention_policy",
        "retention_policy.json",
        "retention_policy.csv"
    )

    # Rerun the Monte Carlo simulations
    rerun_monte_carlo(scenario_name, num_processes, run_numbers, retention_policy)