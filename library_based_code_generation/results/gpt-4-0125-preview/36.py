```python
import matplotlib.pyplot as plt
from Basilisk.utilities.MonteCarlo import Controller, RetentionPolicy, Dispersions
import scenario_AttFeedback

def displayPlots(data, retentionPolicy):
    time = data['time']
    states = data['states']
    plt.figure()
    for state in states:
        plt.plot(time, state)
    plt.xlabel('Time')
    plt.ylabel('States')
    plt.show()

def run(show_plots):
    mc_controller = Controller()
    mc_controller.setSimulationFunction(scenario_AttFeedback.simulationFunction)
    mc_controller.setExecutionFunction(scenario_AttFeedback.executionFunction)
    mc_controller.executionCount = 100
    mc_controller.archiveDir = "mc_archive"
    mc_controller.seedDispersion = True
    mc_controller.threadCount = 4
    mc_controller.verbosity = 2
    mc_controller.variableCasting = True
    mc_controller.dispersionMagnitudeFile = "dispersionMagnitudes.json"

    dispersions = [
        Dispersions.UniformDispersion("parameter1", -1, 1),
        Dispersions.NormalDispersion("parameter2", 0, 0.1),
        Dispersions.ListDispersion("parameter3", [0, 1, 2, 3])
    ]
    for dispersion in dispersions:
        mc_controller.addDispersion(dispersion)

    retention_policy = RetentionPolicy()
    retention_policy.addMessageLogs("*.log")
    retention_policy.setDataCallback(displayPlots if show_plots else None)
    mc_controller.addRetentionPolicy(retention_policy)

    mc_controller.executeSimulations()
    mc_controller.executeCallbacks()

if __name__ == "__main__":
    run(True)
```