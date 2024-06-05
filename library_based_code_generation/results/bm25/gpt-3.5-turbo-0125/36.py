import Basilisk.utilities.MonteCarlo.Controller as Controller
import Basilisk.utilities.MonteCarlo.RetentionPolicy as RetentionPolicy
from Basilisk.utilities.MonteCarlo import Dispersions
import scenario_AttFeedback

def run(show_plots):
    monteCarloController = Controller()
    monteCarloController.setSimulationFunction(test_MonteCarloSimulation)
    monteCarloController.setExecutionFunction(test_MonteCarloSimulation)
    monteCarloController.setExecutionCount(100)
    monteCarloController.setArchiveDirectory("archive")
    monteCarloController.setSeedDispersion(Dispersions.AStar)
    monteCarloController.setThreadCount(4)
    monteCarloController.setVerbosity(2)
    monteCarloController.setVariableCasting(True)
    monteCarloController.setDispersionMagnitudeFile("dispersion_magnitude.txt")
    
    dispersions = [Dispersions.boxAndWingParameters, Dispersions.SCConnectedAndUnconnected]
    monteCarloController.addDispersions(dispersions)
    
    retentionPolicy = RetentionPolicy()
    retentionPolicy.addMessageLogs()
    retentionPolicy.setDataCallback(parseAndLoadXML)
    monteCarloController.addRetentionPolicy(retentionPolicy)
    
    monteCarloController.executeSimulations()
    
    if show_plots:
        displayPlots(data, retentionPolicy)
    
    return

def displayPlots(data, retentionPolicy):
    time, states = pull_and_format_df(data)
    # Plot states against time
    return

if __name__ == "__main__":
    run(True)