```python
import DREAM.Settings.Equations.RunawayElectronDistribution as RED
import DREAM.Settings.Equations.HotElectronDistribution as HED
import DREAM.Settings.Solver as Solver
import DREAM.Settings.Equations.ColdElectronTemperature as CET
import DREAM.Settings.Equations.ElectricField as EF
import DREAM.Settings.Equations.ElectronDensity as ED
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.TransportSettings as Transport
import DREAM
import numpy as np

ds = DREAM.Settings.DREAMSettings()

# Electric field, electron density, and temperature
E_field = 0.5  # V/m
n_e = 1e20  # m^-3
T_e = 1e3  # eV

ds.eqsys.E_field.setPrescribedData(E_field)
ds.eqsys.n_cold.setPrescribedData(n_e)
ds.eqsys.T_cold.setPrescribedData(T_e)

# Momentum grid
pMax = 2.0
n_p = 100
nxi = 10
ds.hottailgrid.setNxi(nxi)
ds.hottailgrid.setNp(n_p)
ds.hottailgrid.setPmax(pMax)

# Initial hot electron Maxwellian
ds.eqsys.f_hot.setInitialProfiles(isotropic=True, T0=T_e)

# Include Dreicer and avalanche
ds.eqsys.n_re.setAvalanche(RED.AVALANCHE_MODE_FLUID)
ds.eqsys.n_re.setDreicer(RED.DREICER_RATE_NEURAL_NETWORK)

# Radial grid
n_r = 10
rMax = 2.0
ds.radialgrid.setBUniform(n_r, rMax)

# Disable runaway grid
ds.runawaygrid.setEnabled(False)

# Transport settings
ds.eqsys.f_hot.transport.setSvensson(T_e)

# Nonlinear solver
ds.solver.setType(Solver.NONLINEAR)

# Time stepper
tMax = 1e-3
n_t = 100
ds.timestep.setTmax(tMax)
ds.timestep.setNt(n_t)

# Save settings to HDF5
ds.save('dream_settings.h5')
```