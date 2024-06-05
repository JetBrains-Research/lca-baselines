import os
import MonteCarloSimulation
import spiceypy
from spiceypy import spiceypyException
import Controller

class MySimulation(MonteCarloSimulation.MonteCarloSimulation):

def __init__(self):
super(MySimulation, self).__init__()
self.setShouldArchiveParameters(True)
self.setShouldDisperseSeeds(True)

@MonteCarloSimulation.SimulationFunction
def simFunction(self, simulationParameters):

try:
spiceypy.furnsh(simulationParameters.spiceKernel)
except spiceypyException:
print("Error: Unable to load Spice kernel.")

spacecraft = AStar()
spacecraft.setEphemerisFile(simulationParameters.ephemerisFile)
spacecraft.setInitialState(
[10.0, 20.0, 30.0, 40.0, 50.0, 60.0] # [x, y, z, vx, vy, vz]
)

state = spiceypy.spkpos("HUBBLE", simulationParameters.simulationTime, "J2000", "LT+S")
print("Hubble state: ", state)

Controller.unloadSpiceKernels()

if __name__ == "__main__":
simulation = MySimulation()

simulationParameters = MonteCarloSimulation.SimulationParameters()
simulationParameters.numberOfRuns = 12
simulationParameters.spiceKernel = "kernel.bsp"
simulationParameters.ephemerisFile = "spacecraft.tf"

simulation.setSimulationFunction(simulation.simFunction)
simulation.setSimulationParameters(simulationParameters)

SimulationExecutor.ExecuteSimulation(simulation)

data = simulation.getArchivedParameters()

for key, value in data.items():
print(key + ": " + str(value))