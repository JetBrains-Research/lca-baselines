 ```python
import matplotlib.pyplot as plt
from Basilisk.utilities.MonteCarlo import Controller, RetentionPolicy, Dispersions
import scenario_AttFeedback

def run(show_plots):
    # Create Monte Carlo simulation controller
    mc = Controller()

    # Set simulation and execution functions
    mc.setSimulationFunction(scenario_AttFeedback.simulation)
    mc.setExecutionFunction(scenario_AttFeedback.execution)

    # Set execution count
    mc.setExecutionCount(1)

    # Set archive directory
    mc.setArchiveDirectory('archive')

    # Set seed dispersion
    mc.setSeedDispersion(0)

    # Set thread count
    mc.setThreadCount(1)

    # Set verbosity
    mc.setVerbosity('HIGH')

    # Set variable casting
    mc.setVariableCasting('double')

    # Set dispersion magnitude file
    mc.setDispersionMagnitudeFile('dispersions.txt')

    # Define list of dispersions
    dispersions = [Dispersions.Constant, Dispersions.Gaussian, Dispersions.Uniform]

    # Add dispersions to Monte Carlo controller
    mc.addDispersions(dispersions)

    # Create retention policy
    retentionPolicy = RetentionPolicy()

    # Add message logs to retention policy
    retentionPolicy.addMessageLog('log1')
    retentionPolicy.addMessageLog('log2')

    # Set data callback
    def dataCallback(data):
        retentionPolicy.retain(data)

    mc.setDataCallback(dataCallback)

    # Add retention policy to Monte Carlo controller
    mc.addRetentionPolicy(retentionPolicy)

    # Execute simulations
    mc.execute()

    # Execute callbacks
    if show_plots:
        displayPlots(mc.retrieve(), retentionPolicy)

def displayPlots(data, retentionPolicy):
    time = data['time']
    states = data['states']

    plt.figure()
    for i in range(states.shape[0]):
        plt.plot(time, states[i, :])
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.title('States vs Time')
    plt.legend([f'State {i}' for i in range(states.shape[0])])
    plt.grid()
    plt.show()

if __name__ == '__main__':
    run(True)
```