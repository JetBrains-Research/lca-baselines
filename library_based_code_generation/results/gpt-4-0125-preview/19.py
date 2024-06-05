```python
import DREAM.Settings.Equations.Runaways as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TimeStepper as TimeStepper
import DREAM

ds = DREAM.Settings.DREAMSettings()

# Physical parameters
E_field = 6  # Electric field strength in V/m
n_e = 5e19  # Electron density in m^-3
T_e = 100  # Temperature in eV

# Grid parameters
pMax = 1  # Maximum momentum in m_e*c
np = 300  # Number of momentum grid points
nxi = 20  # Number of pitch grid points
tMax = 1e-3  # Simulation time in seconds
nt = 20  # Number of time steps

# Set up the grid
ds.hottailgrid.setNxi(nxi)
ds.hottailgrid.setNp(np)
ds.hottailgrid.setPmax(pMax)

ds.radialgrid.setB0(5)  # Set a dummy magnetic field strength to initialize the radial grid

ds.timestep.setTmax(tMax)
ds.timestep.setNt(nt)

# Set constant electric field, density, and temperature
ds.eqsys.E_field.setPrescribedData(E_field)
ds.eqsys.n_i.setPrescribedData(n_e)
ds.eqsys.n_re.setPrescribedData(0)  # Initially no runaway electrons
ds.eqsys.T_cold.setPrescribedData(T_e)

# Set the solver type and time stepper
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.timestep.setType(TimeStepper.TYPE_ADAPTIVE)

# Run the simulation
do = DREAM.runiface(ds, 'output.h5')
```