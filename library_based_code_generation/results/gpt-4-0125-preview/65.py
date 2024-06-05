```python
import numpy as np
import DREAM.Settings.Equations.Ions as Ions
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TransportSettings as Transport
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.TimeStepper as TimeStepper
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.ColdElectronTemperature as ColdElectronTemperature
import DREAM

# Simulation parameters
n_re = 1e20  # Initial number of runaway electrons
T_initial = 10  # Initial temperature in keV
T_final = 1000  # Final temperature in keV
time_steps = 100  # Number of time steps
ion_species = 'D'  # Deuterium ions
E_field = 0.1  # Electric field in V/m
T_cold = 10  # Cold electron temperature in eV
radial_points = 100  # Number of radial points
transport_mode = 'anomalous'  # 'classical' or 'anomalous'
enable_hot_tail = True  # Enable hot tail grid

# Create DREAM settings object
ds = DREAM.Settings.DREAMSettings()

# Radial grid
ds.radialgrid.setB0(5)  # Magnetic field strength in Tesla
ds.radialgrid.setMinorRadius(0.1)  # Minor radius in meters
ds.radialgrid.setNr(radial_points)  # Number of radial points

# Time stepper
ds.timestep.setTmax(T_final)  # Final time in seconds
ds.timestep.setNt(time_steps)  # Number of time steps

# Ions
ds.eqsys.n_i.addIon(name=ion_species, Z=1, n=1e20, iontype=Ions.IONS_DYNAMIC)

# Electric field
ds.eqsys.E_field.setPrescribedData(E_field)

# Cold electron temperature
ds.eqsys.T_cold.setPrescribedData(T_cold)

# Enable hot tail grid
if enable_hot_tail:
    ds.hottailgrid.setEnabled(True)

# Transport settings
if transport_mode == 'classical':
    ds.transport.setMagneticPerturbation(1e-4)
elif transport_mode == 'anomalous':
    ds.transport.setAnomalous(1, 0.1, 0.2)

# Run simulation
do = DREAM.runiface(ds, 'output.h5', quiet=False)
```