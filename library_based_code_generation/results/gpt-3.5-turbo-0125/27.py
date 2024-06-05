import numpy as np
import matplotlib.pyplot as plt
import dream
from dream.simulation import run
from dream.parameters import Settings, Constants
from dream.grid import Grid
from dream.initialization import init
from dream.state import State
from dream.ions import DREAMSettings, Ions
from dream.temperature import Temperature
from dream.radialgrid import RadialGrid
from dream.runawaygrid import RunawayGrid
from dream.hotTailGrid import HotTailGrid

# Set up radial grid
radius = np.linspace(0, 1, 100)
grid = Grid(radial=r)

# Set time steps
dt = 1e-6

# Add ions
ions = Ions(grid)
ions.addCharge(6)
ions.addZ(1)

# Set up temperature and electric field
Te = 1e3
E = 1e3

# Disable runaway and hot-tail grid
runawayGrid = RunawayGrid(grid)
hotTailGrid = HotTailGrid(grid)

# Initialization stage
settings = Settings(grid)
state = State(grid)
init(state, settings, ions, Te, E)

# Save settings
settings.save('settings_init.h5')

# Run simulation
run(state, settings, ions, dt)

# Ionization stage
DREAMSettings.setPreIonization(True)

# Save settings
settings.save('settings_ionization.h5')

# Run simulation
run(state, settings, ions, dt)

# Equilibration stage
DREAMSettings.setPreIonization(False)

# Save settings
settings.save('settings_equilibration.h5')

# Run simulation
run(state, settings, ions, dt)

# Radiation stage
DREAMSettings.setRadiation(True)

# Save settings
settings.save('settings_radiation.h5')

# Run simulation
run(state, settings, ions, dt)

# Plot results
plt.plot(grid.r, state.ni)
plt.plot(grid.r, state.ne)
plt.xlabel('Radius')
plt.ylabel('Density')
plt.legend(['Ion density', 'Electron density'])
plt.show()