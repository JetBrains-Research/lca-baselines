```python
import numpy as np
import matplotlib.pyplot as plt
from DREAM.DREAMSettings import DREAMSettings
from DREAM import runiface
from DREAM.Settings.Equations.ElectricField import ElectricField
from DREAM.Settings.Equations.IonSpecies import IonSpecies
from DREAM.Settings.RadialGrid import RadialGrid
from DREAM.Settings.Solver import Solver

# Initialization
ds = DREAMSettings()

# Radial grid
n_r = 10
r_min = 0.0
r_max = 1.0
ds.radialgrid.setB0(5, r0=1.5)
ds.radialgrid.setMinorRadius(0.5)
ds.radialgrid.setMajorRadius(1.5)
ds.radialgrid.setWallRadius(1.6)
ds.radialgrid.setType(RadialGrid.TYPE_UNIFORM)
ds.radialgrid.setNr(n_r)

# Time steps
t_max = 1e-3  # Maximum simulation time in seconds
n_t = 100  # Number of time steps
ds.timestep.setTmax(t_max)
ds.timestep.setNt(n_t)

# Ions
ds.eqsys.n_i.addIon(name='D', Z=1, iontype=IonSpecies.IONS_DYNAMIC, n=1e20, r=np.linspace(r_min, r_max, n_r))

# Temperature and electric field
ds.eqsys.T_cold.setPrescribedData(1e3, times=np.linspace(0, t_max, n_t), radius=np.linspace(r_min, r_max, n_r))
ds.eqsys.E_field.setPrescribedData(0.1, times=np.linspace(0, t_max, n_t), radius=np.linspace(r_min, r_max, n_r))

# Disable runaway and hot-tail grid
ds.hottailgrid.setEnabled(False)
ds.runawaygrid.setEnabled(False)

# Solver settings
ds.solver.setType(Solver.LINEAR_IMPLICIT)  # Use a linear implicit solver

# Save settings after initialization
ds.save('settings_init.h5')

# Ionization stage
ds.eqsys.n_i.setIonization(IonSpecies.IONIZATION_MODE_FLUID)
ds.save('settings_ionization.h5')

# Equilibration stage
# No additional settings needed for this simple example, but this is where you'd adjust for equilibration if needed.

# Radiation stage
ds.eqsys.radiation.setSynchrotron(True)
ds.eqsys.radiation.setBremsstrahlung(True)
ds.eqsys.radiation.setCyclotron(True)
ds.save('settings_radiation.h5')

# Run simulation
do = runiface(ds, 'output.h5', quiet=False)

# Plotting results
fig, ax = plt.subplots(1, 1)
ax.plot(do.grid.t, do.eqsys.T_cold.data, label='Temperature')
ax.plot(do.grid.t, do.eqsys.E_field.data, label='Electric Field')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (eV) / Electric Field (V/m)')
ax.legend()
plt.show()

# Note: This code does not directly plot ohmic heating and radiative losses as the DREAM library API specifics for accessing these quantities directly were not provided. Users should refer to the DREAM documentation for methods to extract and plot these specific quantities.
```