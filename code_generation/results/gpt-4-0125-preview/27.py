```python
import numpy as np
import matplotlib.pyplot as plt
from DREAM.DREAMSettings import DREAMSettings
from DREAM import runiface
from DREAM.Settings.Equations import Ions

# Stage 1: Initialization
ds = DREAMSettings()

# Radial grid
ds.radialgrid.setB0(5)  # Tesla
ds.radialgrid.setMinorRadius(0.6)  # meters
ds.radialgrid.setMajorRadius(1.8)  # meters
ds.radialgrid.setWallRadius(0.6)  # meters
ds.radialgrid.setNr(10)

# Time steps
ds.timestep.setTmax(1e-3)  # seconds
ds.timestep.setNt(10)

# Ions
ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC, n=1e20)

# Temperature and electric field
ds.eqsys.T_cold.setPrescribedData(1e3)  # eV
ds.eqsys.E_field.setPrescribedData(0.1)  # V/m

# Disable runaway and hot-tail grid
ds.hottailgrid.setEnabled(False)
ds.runawaygrid.setEnabled(False)

# Save settings for initialization stage
ds.save('settings_init.h5')

# Run simulation for initialization stage
runiface(ds, 'output_init.h5', quiet=False)

# Stage 2: Ionization
ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
ds.save('settings_ionization.h5')
runiface(ds, 'output_ionization.h5', quiet=False)

# Stage 3: Equilibration
ds.timestep.setTmax(1e-2)  # Longer time for equilibration
ds.timestep.setNt(100)
ds.save('settings_equilibration.h5')
runiface(ds, 'output_equilibration.h5', quiet=False)

# Stage 4: Radiation
ds.eqsys.n_i.addIon(name='Ar', Z=18, iontype=Ions.IONS_DYNAMIC, n=1e17, ionization=Ions.IONIZATION_MODE_FLUID)
ds.eqsys.radiation.setBremsstrahlung(True)
ds.eqsys.radiation.setSynchrotron(True)
ds.eqsys.radiation.setCyclotron(True)
ds.save('settings_radiation.h5')
runiface(ds, 'output_radiation.h5', quiet=False)

# Plotting results
output_init = runiface.LoadOutput('output_init.h5')
output_ionization = runiface.LoadOutput('output_ionization.h5')
output_equilibration = runiface.LoadOutput('output_equilibration.h5')
output_radiation = runiface.LoadOutput('output_radiation.h5')

temps = output_equilibration.grid.t[1:]
ohmic_heating = output_equilibration.other.fluid.ohmic_heating[:]
radiative_losses = output_radiation.other.fluid.radiation[:]

plt.figure(figsize=(10, 6))
plt.plot(temps, ohmic_heating, label='Ohmic Heating')
plt.plot(temps, radiative_losses, label='Radiative Losses')
plt.xlabel('Temperature (eV)')
plt.ylabel('Power Density (W/m^3)')
plt.title('Energy Balance in Plasma')
plt.legend()
plt.show()
```