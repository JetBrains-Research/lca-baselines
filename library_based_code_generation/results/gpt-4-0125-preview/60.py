```python
import basilisk.utilities.MonteCarloSupport as MCS
import multiprocessing
import sys

def rerun_monte_carlo(scenario_name, num_processes, run_numbers, retention_policy=None):
    mc_controller = MCS.MonteCarloController()
    
    # Set the scenario name
    mc_controller.scenarioName = scenario_name
    
    # Set the initial conditions and archive directories
    mc_controller.setInitialConditionsDirectory(f"./{scenario_name}/InitialConditions")
    mc_controller.setArchiveDirectory(f"./{scenario_name}/Archive")
    
    # Set the execution count and whether to disperse seeds or archive parameters
    mc_controller.executionCount = len(run_numbers)
    mc_controller.disperseSeeds = True
    mc_controller.archiveParams = True
    
    # Add the specified retention policy if provided
    if retention_policy:
        mc_controller.addRetentionPolicy(retention_policy)
    
    # Set the number of processes to spawn
    if num_processes > 1:
        mc_controller.setThreadCount(num_processes)
    else:
        mc_controller.setThreadCount(multiprocessing.cpu_count())
    
    # Specify the run numbers to rerun
    mc_controller.specifyRunNumbers(run_numbers)
    
    # Run the initial conditions
    mc_controller.runAll()

if __name__ == "__main__":
    # Example usage
    scenario = "ExampleScenario"
    processes = 4
    runs_to_rerun = [1, 2, 3]
    rerun_monte_carlo(scenario, processes, runs_to_rerun)
```