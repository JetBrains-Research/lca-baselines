```python
import DREAM.Settings.Equations.ColdElectronTemperature as T_cold
import DREAM.Settings.Equations.ElectricField as E_field
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TimeStepper as TimeStepper
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM
import numpy as np

# Initial setup
ds = DREAM.Settings.DREAMSettings()

# Radial grid
ds.radialgrid.setB0(5)  # Tesla
ds.radialgrid.setMinorRadius(0.22)  # meters
ds.radialgrid.setMajorRadius(0.68)  # meters
ds.radialgrid.setWallRadius(0.24)  # meters
ds.radialgrid.setNr(10)

# Time stepper
ds.timestep.setTmax(1e-3)  # seconds
ds.timestep.setNt(10)

# Ions
ds.eqsys.n_i.addIon(name='D', Z=1, iontype=DREAM.Settings.Equations.Ions.IONS_DYNAMIC, n=1e20, r=np.linspace(0, 0.22, 10))

# Electric field
ds.eqsys.E_field.setType(E_field.TYPE_SELFCONSISTENT)

# Cold electron temperature
ds.eqsys.T_cold.setType(T_cold.TYPE_SELFCONSISTENT)

# Hot tail grid (disable)
ds.hottailgrid.setEnabled(False)

# Runaway grid (disable)
ds.runawaygrid.setEnabled(False)

# Solver settings
ds.solver.setType(Solver.LINEAR_IMPLICIT)  # Use a linear solver
ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
ds.solver.setTolerance(1e-5)
ds.solver.setMaxIterations(100)

# Save settings to HDF5 file
ds.save('settings1.h5')

# Run simulation
do = DREAM.runiface(ds, 'output1.h5')

# Restart simulation twice
for i in range(2):
    # Load settings from previous run
    ds = DREAM.Settings.DREAMSettings('output{}.h5'.format(i+1))
    
    # Adjust time stepper for the restart
    ds.timestep.setTmax(ds.timestep.tmax + 1e-3)  # Increase Tmax
    ds.timestep.setNt(10)  # Reset number of time steps
    
    # Save settings to new HDF5 file
    ds.save('settings{}.h5'.format(i+2))
    
    # Run simulation
    do = DREAM.runiface(ds, 'output{}.h5'.format(i+2))
```