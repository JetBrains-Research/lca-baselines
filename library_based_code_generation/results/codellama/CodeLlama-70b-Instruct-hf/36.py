



 Here is the Python code that meets the requirements:

```
import numpy as np
import matplotlib.pyplot as plt
from Basilisk.utilities.MonteCarlo.Controller import Controller, RetentionPolicy, MonteCarloException
from Basilisk.utilities.MonteCarlo.Dispersion import UniformEulerAngleMRPDispersion, UniformDispersion, NormalVectorCartDispersion
from Basilisk.simulation import sim_model
import scenario_AttFeedback

def run(show_plots):
    #   Create a Monte Carlo simulation controller
    monteCarlo = Controller()

    #   Define simulation and execution functions
    def simulationFunction(monteCarlo):
        scenario_AttFeedback.run(
            show_plots=False,
            useJitter=True,
            useRWVoltageIO=True
        )

    def executeCallbacks(data, retentionPolicy):
        if show_plots:
            displayPlots(data, retentionPolicy)

    #   Set simulation and execution functions
    monteCarlo.setSimulationFunction(simulationFunction)
    monteCarlo.setExecutionCallback(executeCallbacks)

    #   Set execution count, archive directory, and seed dispersion
    monteCarlo.setExecutionCount(1000)
    monteCarlo.setArchiveDir('monteCarloArchive')
    monteCarlo.setSeedDispersion(UniformDispersion(0, 10000))

    #   Set thread count, verbosity, and variable casting
    monteCarlo.setThreadCount(1)
    monteCarlo.setVerbose(True)
    monteCarlo.setVarcast(np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]))

    #   Define a list of dispersions
    dispList = [
        UniformEulerAngleMRPDispersion(-10 * np.pi / 180, 10 * np.pi / 180),
        NormalVectorCartDispersion(np.sqrt(0.2), 0.0, 0.0, 1.0),
        UniformDispersion(-0.2, 0.2)
    ]

    #   Add dispersions to the Monte Carlo controller
    monteCarlo.setDispersionList(dispList)

    #   Create a retention policy
    retentionPolicy = RetentionPolicy()
    retentionPolicy.setDataCallback(monteCarlo.getLogData)

    #   Add message logs to the retention policy
    retentionPolicy.addMessageLog(sim_model.scenarioMessage)
    retentionPolicy.addMessageLog(sim_model.logOutputData)

    #   Add the retention policy to the Monte Carlo controller
    monteCarlo.addRetentionPolicy(retentionPolicy)

    #   Execute the simulations
    monteCarlo.executeSimulations()

    #   Execute callbacks if 'show_plots' is True
    if show_plots:
        monteCarlo.executeCallbacks()

def displayPlots(data, retentionPolicy):
    #   Extract time and states from the data
    timeData = data['attErrorInertial3DMsg.outputDataSigma'][:, 0] * macros.NANO2SEC
    sigma_BR = data['attErrorInertial3DMsg.outputDataSigma'][:, 1:]

    #   Plot the states against time
    plt.figure(1)
    for idx in range(1, 4):
        plt.plot(timeData, sigma_BR[:, idx-1],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='$\sigma_' + str(idx) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Attitude Error $\sigma_{B/R}$')

if __name__ == "__main__":
    run(True)
```

This code defines a function `run` that creates a Monte Carlo simulation controller, sets its simulation and execution functions, execution count, archive directory, seed dispersion, thread count, verbosity, variable casting, and dispersion magnitude file. It also defines a list of dispersions and adds them to the Monte Carlo controller. The function creates a retention policy, adds message logs to it, sets its data callback, and adds it to the Monte Carlo controller. The function executes the simulations, executes callbacks if `show_plots` is True, and returns. The code also defines a function `displayPlots` that takes `data` and `retentionPolicy` as parameters, extracts time and states from the data, and plots the states against time. Finally, it calls the `run` function with `True` as the argument if the script is run as the main program.