```python
import basilisk
from Basilisk import monteCarlo

def rerunMonteCarloSimulations(scenario_name, num_processes, run_numbers, retention_policies):
    monteCarlo.controller = monteCarlo.MonteCarloController()
    monteCarlo.controller.setInitialConditionsDir("initial_conditions")
    monteCarlo.controller.setArchiveDir("archive")
    monteCarlo.controller.setExecutionCount(num_processes)
    monteCarlo.controller.setShouldDisperseSeeds(True)
    monteCarlo.controller.setShouldArchiveParameters(True)
    monteCarlo.controller.addRetentionPoliciesToSim(retention_policies)
    monteCarlo.controller.runInitialConditions(scenario_name)
    
if __name__ == "__main__":
    rerunMonteCarloSimulations("scenario_name", 4, [1, 2, 3], ["retention_policy1", "retention_policy2"])
```