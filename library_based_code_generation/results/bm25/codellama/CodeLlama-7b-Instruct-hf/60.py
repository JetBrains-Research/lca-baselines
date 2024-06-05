  import os
import numpy as np
from basilisk import MonteCarloSimulation

def rerun_monte_carlo_simulations(scenario_name, num_processes, run_numbers, retention_policies):
    # Set up the Monte Carlo controller
    mc_controller = MonteCarloSimulation(scenario_name)

    # Set the initial conditions directory
    mc_controller.setInitialConditionsDirectory("./initial_conditions")

    # Set the archive directory
    mc_controller.setArchiveDirectory("./archive")

    # Set the execution count
    mc_controller.setExecutionCount(100)

    # Set whether to disperse seeds or archive parameters
    mc_controller.setShouldDisperseSeeds(True)
    mc_controller.setShouldArchiveParameters(True)

    # Add the specified retention policies
    for retention_policy in retention_policies:
        mc_controller.addRetentionPolicy(retention_policy)

    # Run the initial conditions
    mc_controller.runInitialConditions()

    # Extract the subset of runs
    runs = mc_controller.extractSubsetOfRuns(run_numbers)

    # Add the retention function
    for run in runs:
        mc_controller.addRetentionFunction(run)

    # Run the simulation
    mc_controller.run()

if __name__ == "__main__":
    # Define the scenario name
    scenario_name = "my_scenario"

    # Define the number of processes to spawn
    num_processes = 4

    # Define the run numbers to rerun
    run_numbers = [1, 2, 3, 4]

    # Define the retention policies
    retention_policies = [RetentionPolicy("my_retention_policy", 0.5)]

    # Rerun the Monte Carlo simulations
    rerun_monte_carlo_simulations(scenario_name, num_processes, run_numbers, retention_policies)