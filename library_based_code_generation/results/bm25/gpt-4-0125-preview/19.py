```python
import DREAM.Settings.Equations.RunawayElectronDensity as RED
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TimeStepper as TimeStepper
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Atomics as Atomics
import DREAM
import numpy as np

ds = DREAM.Settings.DREAMSettings()

# Physical parameters
E_field = 6  # Electric field strength in V/m
n_e = 5e19  # Electron density in m^-3
T_e = 100  # Temperature in eV

# Grid parameters
pMax = 1  # Maximum momentum in m_e*c
n_p = 300  # Number of momentum grid points
nXi = 20  # Number of pitch grid points
tMax = 1e-3  # Simulation time in seconds
n_t = 20  # Number of time steps

# Set up the grid
ds.grid.setMomentumGrid(DREAM.Settings.Grid.MomentumGrid(pMax=pMax, np=n_p))
ds.grid.setPitchGrid(nXi=nXi)
ds.grid.setRadialGrid(n_r=1)
ds.grid.setTimeGrid(tMax=tMax, nt=n_t)

# Set up the physical parameters
ds.eqsys.E_field.setPrescribedData(E_field)
ds.eqsys.n_i.setPrescribedData(density=n_e, Z=1, ion="D")
ds.eqsys.n_re.setAvalanche(RED.AVALANCHE_MODE_FLUID_HESSLOW)
ds.eqsys.n_re.setDreicer(RED.DREICER_RATE_NEURAL_NETWORK)

# Set up the temperature
ds.eqsys.T_cold.setPrescribedData(T_e)

# Solver settings
ds.solver.setType(Solver.NONLINEAR)
ds.solver.tolerance.set(reltol=1e-5)
ds.solver.setMaxIterations(100)
ds.solver.setVerbose(True)

# Time stepper settings
ds.timestep.setTmax(tMax)
ds.timestep.setNt(n_t)

# Run the simulation
do = DREAM.runiface(ds, 'output.h5')
```