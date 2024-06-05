import Basilisk.utilities.MonteCarlo.Controller as Controller
import Basilisk.utilities.MonteCarlo.RetentionPolicy as RetentionPolicy
from Basilisk.utilities.MonteCarlo import Dispersions
import scenario_AttFeedback

def run(show_plots):
    monteCarloController = Controller()
    monteCarloController.setSimulationFunction(simulation_function)
    monteCarloController.setExecutionFunction(execution_function)
    monteCarloController.setExecutionCount(100)
    monteCarloController.setArchiveDirectory("archive_directory")
    monteCarloController.setSeedDispersion(seed_dispersion)
    monteCarloController.setThreadCount(4)
    monteCarloController.setVerbosity(2)
    monteCarloController.setVariableCasting(True)
    monteCarloController.setDispersionMagnitudeFile("dispersion_magnitude_file")

    dispersions = [Dispersions.Dispersion1, Dispersions.Dispersion2, Dispersions.Dispersion3]
    for dispersion in dispersions:
        monteCarloController.addDispersion(dispersion)

    retentionPolicy = RetentionPolicy()
    retentionPolicy.addMessageLogs(message_logs)
    retentionPolicy.setDataCallback(data_callback)
    monteCarloController.addRetentionPolicy(retentionPolicy)

    monteCarloController.executeSimulations()

    if show_plots:
        displayPlots(data, retentionPolicy)

    return

def displayPlots(data, retentionPolicy):
    time = extractTime(data)
    states = extractStates(data)
    plotStatesAgainstTime(states, time)

if __name__ == "__main__":
    run(True)