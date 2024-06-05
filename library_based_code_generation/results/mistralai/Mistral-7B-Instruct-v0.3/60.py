 Here is a Python script that follows your instructions, using the Basilisk library. Please note that you may need to adjust the code to fit your specific project requirements.

```python
import os
import sys
import multiprocessing
import basilisk.utilities.ephemeris as basilisk_ephemeris
import basilisk.utilities.io_utils as basilisk_io
import basilisk.utilities.constants as basilisk_constants
import basilisk.fsw_simulation.fsw_simulation as basilisk_fsw

def rerun_monte_carlo_simulations(scenario_name, num_processes, run_numbers_to_rerun, retention_policy, initial_conditions_dir, archive_dir, exec_count, disperse_seeds, archive_params):
    basilisk_io.set_output_verbosity(basilisk_io.Verbosity.terse)

    # Set up the Monte Carlo controller
    mc_controller = basilisk_fsw.MonteCarloController()

    # Specify the initial conditions directory
    mc_controller.set_initial_conditions_directory(initial_conditions_dir)

    # Specify the archive directory
    mc_controller.set_archive_directory(archive_dir)

    # Specify the number of executions
    mc_controller.set_execution_count(exec_count)

    # Specify whether to disperse seeds or archive parameters
    mc_controller.set_disperse_seeds(disperse_seeds)
    mc_controller.set_archive_parameters(archive_params)

    # Add the specified retention policy
    mc_controller.add_retention_policy(retention_policy)

    # Run the initial conditions
    mc_controller.run_initial_conditions()

    # If the function is run as the main program, call itself
    if __name__ == "__main__":
        pool = multiprocessing.Pool(processes=num_processes)
        pool.map(rerun_monte_carlo_simulations, zip([scenario_name] * len(run_numbers_to_rerun), [num_processes] * len(run_numbers_to_rerun), run_numbers_to_rerun, [retention_policy] * len(run_numbers_to_rerun), [initial_conditions_dir] * len(run_numbers_to_rerun), [archive_dir] * len(run_numbers_to_rerun), [exec_count] * len(run_numbers_to_rerun), [disperse_seeds] * len(run_numbers_to_rerun), [archive_params] * len(run_numbers_to_rerun)))

if __name__ == "__main__":
    scenario_name = "my_scenario"
    num_processes = 4
    run_numbers_to_rerun = [1, 3, 5, 7]
    retention_policy = basilisk_fsw.RetentionPolicy.archive_all
    initial_conditions_dir = "initial_conditions"
    archive_dir = "archives"
    exec_count = 10
    disperse_seeds = False
    archive_params = False
    rerun_monte_carlo_simulations(scenario_name, num_processes, run_numbers_to_rerun, retention_policy, initial_conditions_dir, archive_dir, exec_count, disperse_seeds, archive_params)
```

This script sets up a Monte Carlo simulation using the Basilisk library and reruns a specified number of runs based on the provided parameters. It also allows for the addition of a new retention policy and can be run in parallel using multiple processes.