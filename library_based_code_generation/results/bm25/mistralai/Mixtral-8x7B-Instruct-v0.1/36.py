 ```python
import Controller
from Basilisk.utilities.MonteCarlo import RetentionPolicy, Dispersions, AStar, boxAndWingParameters, \
    __createMessageAndHeaderTemplate, setShouldArchiveParameters, setShouldDisperseSeeds, test_MonteCarloSimulation, \
    eigAxisAndAngleToDCM, seperateFilesAndDirs, SCConnectedAndUnconnected, pull_and_format_df, SCTransAndRotation, \
    parseAndLoadXML, asEigen, effortBasedAStar, test_MonteCarloSimulationDatashader, extForceInertialAndTorque, \
    extForceBodyAndTorque

def run(show_plots):
    # Create Monte Carlo simulation controller
    monteCarloController = Controller.Controller()

    # Set simulation and execution functions
    monteCarloController.setSimulationFunction(test_MonteCarloSimulation)
    monteCarloController.setExecutionFunction(test_MonteCarloSimulationDatashader)

    # Set execution count
    monteCarloController.setExecutionCount(1)

    # Set archive directory
    monteCarloController.setArchiveDirectory('archive')

    # Set seed dispersion
    monteCarloController.setShouldDisperseSeeds(True)
    monteCarloController.setSeedDispersion(Dispersions.DISP_UNIFORM)

    # Set thread count
    monteCarloController.setThreadCount(1)

    # Set verbosity
    monteCarloController.setVerbosity(1)

    # Set variable casting
    monteCarloController.setVariableCasting('double')

    # Set dispersion magnitude file
    monteCarloController.setDispersionMagnitudeFile('dispersions.dat')

    # Define list of dispersions
    dispersions = [Dispersions.DISP_UNIFORM, Dispersions.DISP_GAUSSIAN, Dispersions.DISP_SINUSOIDAL]

    # Add dispersions to Monte Carlo controller
    monteCarloController.addDispersions(dispersions)

    # Create retention policy
    retentionPolicy = RetentionPolicy.RetentionPolicy()

    # Add message logs to retention policy
    retentionPolicy.addMessageLog('logs.txt')

    # Set data callback
    def dataCallback(data):
        print(data)
    retentionPolicy.setDataCallback(dataCallback)

    # Add retention policy to Monte Carlo controller
    monteCarloController.addRetentionPolicy(retentionPolicy)

    # Execute simulations
    monteCarloController.execute()

    # Execute callbacks
    if show_plots:
        data, retentionPolicy = monteCarloController.getAccumulatedData()
        displayPlots(data, retentionPolicy)

def displayPlots(data, retentionPolicy):
    time = data['time']
    states = data['states']

    # Plot states against time
    # (Replace the following line with your desired plotting code)
    pass

if __name__ == '__main__':
    run(True)
```