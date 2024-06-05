```python
import numpy as np
import DREAM.Settings.Equations.Ions as Ions
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TransportSettings as TransportSettings
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.TimeStepper as TimeStepper
import DREAM.Settings.ColdElectronTemperature as ColdElectronTemperature
import DREAM

# Simulation parameters
initial_temperature = 10  # in eV
final_temperature = 1000  # in eV
time_steps = 100
ion_species = 'D'
electric_field = 0.1  # in V/m
cold_electron_temperature = 20  # in eV
radial_grid_points = 100
transport_modes = ['classical', 'neo-classical', 'turbulent']
hot_tail_grid_enabled = True

# Create DREAM settings object
ds = DREAM.Settings.DREAMSettings()

# Radial grid
ds.radialgrid.setB0(5)  # Magnetic field strength in Tesla
ds.radialgrid.setMinorRadius(0.1)  # Minor radius in meters
ds.radialgrid.setMajorRadius(1.5)  # Major radius in meters
ds.radialgrid.setWallRadius(0.15)  # Wall radius in meters
ds.radialgrid.setNr(radial_grid_points)

# Time stepper
ds.timestep.setTmax(1e-3)  # Max simulation time in seconds
ds.timestep.setNt(time_steps)

# Ions
ds.eqsys.n_i.addIon(name=ion_species, Z=1, n=1e20, iontype=Ions.IONS_DYNAMIC)

# Electric field and cold electron temperature
ds.eqsys.E_field.setPrescribedData(electric_field)
ds.eqsys.T_cold.setPrescribedData(cold_electron_temperature)

# Enable hot tail grid if required
if hot_tail_grid_enabled:
    ds.hottailgrid.setEnabled(True)

# Transport settings
for mode in transport_modes:
    if mode == 'classical':
        ds.transport.setMagneticPerturbation(1e-4, 1e-2, m=1, n=1)
    elif mode == 'neo-classical':
        ds.transport.setDreicer(TransportSettings.COLLQTY_DREICER_RATE_NEURAL_NETWORK)
    elif mode == 'turbulent':
        ds.transport.setTurbulenceAmplitude(1e-2)

# Run simulation
do = DREAM.runiface(ds, 'output.h5', quiet=False)
```