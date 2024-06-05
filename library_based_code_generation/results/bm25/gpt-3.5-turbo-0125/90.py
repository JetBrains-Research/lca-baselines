import numpy as np
import h5py
import DREAM
from DREAM import runSimulation, TimeStepper, DREAMSettings, setSaveSettings, setGrid, generateSettings, setTemperature, ElectricField

# Set up simulation parameters
tMax = 1e-3
Nt = 1000
Nr = 100
Np = 100
Nz = 100

# Set up radial grid
radius = np.linspace(0, 1, Nr)
psi = np.linspace(0, 1, Np)
zeta = np.linspace(0, 1, Nz)
grid = DREAM.Grid.RadialGrid(radius, psi, zeta)

# Set the time stepper
stepper = TimeStepper.RK4

# Add ions
Z = 1
n = 1e20
T = 100
ions = DREAM.Ions.Hydrogen(Z=Z, n=n, T=T)

# Set the electric field and cold electron temperature
Efield = ElectricField.ColdStart
Te = 10
setTemperature(Te)

# Set up hot tail grid
Nxi = 5
Nxi_hot = 10
grid_hot = DREAM.Grid.HotTailGrid(Nxi, Nxi_hot)

# Disable runaway grid
disableRunawayGrid()

# Set the solver type and its parameters
solver = DREAM.Solver.HPIC
solver.tolerance = 1e-6

# Save settings to HDF5 file and run simulation
setSaveSettings('settings.h5')
runSimulation()

# Restart simulation twice
for i in range(2):
    settings = DREAMSettings('settings.h5')
    settings.tMax = 2e-3
    setSaveSettings('settings_restart{}.h5'.format(i))
    runSimulation()