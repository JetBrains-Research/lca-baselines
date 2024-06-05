```python
from DREAM.DREAMSettings import DREAMSettings
from DREAM import runiface
import numpy as np

# Initial setup
ds = DREAMSettings()

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
ds.eqsys.n_i.addIon(name='D', Z=1, iontype=1, n=1e20)

# Electric field
ds.eqsys.E_field.setPrescribedData(5e6)  # V/m

# Cold electron temperature
ds.eqsys.T_cold.setPrescribedData(10)  # eV

# Hot tail grid (disable kinetic equations)
ds.hottailgrid.setEnabled(False)

# Runaway grid
ds.runawaygrid.setEnabled(False)

# Solver settings
ds.solver.setType(ds.solver.LINEAR_IMPLICIT)  # Use linear solver
ds.solver.setLinearSolver(reltol=1e-5)

# Save settings to HDF5
ds.save('settings1.h5')

# Run simulation
runiface(ds, 'output1.h5', quiet=False)

# Restart simulation, adjusting time stepper
for i in range(2):
    # Load settings from previous run
    ds = DREAMSettings(loadfrom='output{}.h5'.format(i+1))
    
    # Adjust time stepper for restart
    ds.timestep.setTmax(ds.timestep.tmax + 1e-3)  # Increase Tmax
    ds.timestep.setNt(ds.timestep.nt + 10)  # Increase number of time steps
    
    # Save new settings
    ds.save('settings{}.h5'.format(i+2))
    
    # Run simulation
    runiface(ds, 'output{}.h5'.format(i+2), quiet=False)
```