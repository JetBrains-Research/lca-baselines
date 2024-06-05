import dream
import numpy as np

# Set electric field strength, electron density, and temperature
E_field = 1.0
n_e = 1e19
T_e = 1e3

# Define momentum grid
p_grid = dream.Grid.Grid(np.linspace(0, 10, 100))

# Set up initial hot electron Maxwellian
hot_maxwellian = dream.HotElectronDistribution.HotElectronDistribution(p_grid, T_e)

# Include Dreicer and avalanche
dreicer = dream.include.Dreicer()
avalanche = dream.include.Avalanche()

# Set up radial grid
r_grid = dream.RadialGrid.RadialGrid(0, 1, 100)

# Disable runaway grid
runaway_grid = dream.RunawayElectronDensity.RunawayElectronDensity()

# Set Svensson transport coefficients
svensson = dream.include.Svensson()

# Use nonlinear solver
solver = dream.setNonlinearSolver()

# Set time stepper
time_stepper = dream.setTimeStepper()

# Save settings to HDF5 file
dream.saveSettings('simulation_settings.h5')