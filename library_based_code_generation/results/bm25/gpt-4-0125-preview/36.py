```python
from Basilisk.utilities.MonteCarlo import Controller, RetentionPolicy, Dispersions
import scenario_AttFeedback
import matplotlib.pyplot as plt

def run(show_plots):
    mc_controller = Controller()
    mc_controller.setExecutionFunction(scenario_AttFeedback.executionFunction)
    mc_controller.setSimulationFunction(scenario_AttFeedback.simulationFunction)
    mc_controller.executionCount = 100
    mc_controller.archiveDir = "mc_archive"
    mc_controller.setShouldDisperseSeeds(True)
    mc_controller.threadCount = 4
    mc_controller.verbosity = 2
    mc_controller.variableCasting = True
    mc_controller.dispersionMagnitudeFile = "dispersionMagnitudes.json"

    dispersions = [
        Dispersions.AStar(),
        Dispersions.boxAndWingParameters(),
        Dispersions.eigAxisAndAngleToDCM(),
        Dispersions.extForceInertialAndTorque(),
        Dispersions.extForceBodyAndTorque()
    ]
    for dispersion in dispersions:
        mc_controller.addDispersion(dispersion)

    retention_policy = RetentionPolicy()
    retention_policy.addMessageLogs("*.txt")
    retention_policy.setDataCallback(displayPlots if show_plots else None)
    mc_controller.setRetentionPolicy(retention_policy)

    mc_controller.executeSimulations()
    if show_plots:
        mc_controller.executeCallbacks()

def displayPlots(data, retentionPolicy):
    time = data['time']
    states = data['states']
    plt.plot(time, states)
    plt.xlabel('Time')
    plt.ylabel('States')
    plt.title('States vs Time')
    plt.show()

if __name__ == "__main__":
    run(True)
```