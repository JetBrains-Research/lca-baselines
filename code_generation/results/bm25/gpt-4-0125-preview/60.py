```python
import basilisk.simulation as bs
import basilisk.utilities as bu
import basilisk.architecture as ba

def rerun_monte_carlo(scenario_name, num_processes, run_numbers, new_retention_policy=None):
    mc_controller = bs.MonteCarloController()
    mc_controller.setArchiveDir(f"./archive/{scenario_name}")
    mc_controller.setInitialConditionsDir(f"./initial_conditions/{scenario_name}")
    mc_controller.setExecutionCount(len(run_numbers))
    mc_controller.setShouldDisperseSeeds(True)
    mc_controller.setShouldArchiveParameters(True)

    if new_retention_policy:
        retention_policy = ba.RetentionPolicy()
        retention_policy.addRetentionFunction(new_retention_policy)
        mc_controller.addRetentionPolicy(retention_policy)

    subset_of_runs = bu.extractSubsetOfRuns(run_numbers)
    mc_controller.runInitialConditions(subset_of_runs, num_processes)

if __name__ == "__main__":
    scenario_name = "example_scenario"
    num_processes = 4
    run_numbers = [1, 2, 3, 4]
    new_retention_policy = lambda data: data.getNumOfDevices() > 2
    rerun_monte_carlo(scenario_name, num_processes, run_numbers, new_retention_policy)
```