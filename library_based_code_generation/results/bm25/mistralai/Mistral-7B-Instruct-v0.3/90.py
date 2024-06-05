 Here is a Python script that sets up a self-consistent fluid DREAM run as per your instructions. Please note that this script assumes you have the DREAM library installed and properly configured.

```python
import os
from dream import TimeStepper, DREAMSettings, ElectricField, runSimulation, loadCODE
from dream.units import getTimeAndUnit, formatTimeAndUnit
from dream.grids import generateGrid
from dream.utilities import tBeforeOnsetFromQCritAndTcoldFromOutput, tBeforeOnsetFromQCritAndPelletShardPosition

# Set up simulation parameters
settings = DREAMSettings()
settings.setGrid(generateGrid(100, 1e-3, 1e-6))  # Radial grid
settings.setTimeStepper(TimeStepper('implicit-euler'))  # Time stepper
settings.setTemperature(ElectricField(0.0, 0.0), 1e4)  # Electric field and cold electron temperature
settings.setIons('D', 1e14)  # Ions
settings.setHotTailGrid(False)  # Disable runaway grid
settings.setSolverType('runTE')  # Solver type
settings.setSaveSettings(True)  # Save settings to HDF5 file

# Save settings to an HDF5 file
settings.save('dream_settings.h5')

# Run the initial simulation
runSimulation('dream_settings.h5')

# Restart the simulation twice
for i in range(1, 4):
    # Load settings from the output of the previous run
    settings = loadCODE('dream_output_%i.h5' % (i - 1))

    # Adjust the time stepper
    settings.getTimeStepper().setCFL(0.5)

    # Save the new settings to an HDF5 file
    settings.save('dream_settings_restart_%i.h5' % i)

    # Run the simulation again
    runSimulation('dream_settings_restart_%i.h5' % i)
```

This script sets up a self-consistent fluid DREAM run with the specified parameters, saves the settings to an HDF5 file, runs the initial simulation, and then restarts the simulation twice, each time loading the settings from the output of the previous run, adjusting the time stepper, and saving the new settings to an HDF5 file before running the simulation again.