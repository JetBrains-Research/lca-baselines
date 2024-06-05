import numpy as np
import matplotlib.pyplot as plt
import h5py

from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAM import DREAM

ds = DREAMSettings()

# Set up simulation parameters
ds.collisions.collfreq_type = DREAM.COLLISION_FREQ_TYPE_FULL
ds.eqsys.n_i.addFluid(n=1e20, r=0, t=5)
ds.eqsys.n_cold.setPrescribedData(1e20)

# Set up radial grid
ds.radialgrid.setB0(5)
ds.radialgrid.setMinorRadius(0.5)
ds.radialgrid.setWallRadius(2)

# Set the time stepper
ds.timestep.setTmax(1e-3)
ds.timestep.setNt(1000)

# Add ions
ds.eqsys.n_i.addIon(name='D', Z=1, iontype=DREAM.IONS_DYNAMIC_FULLY_IONIZED)

# Set electric field and cold electron temperature
ds.eqsys.E_field.setPrescribedData(0)
ds.eqsys.T_cold.setPrescribedData(5)

# Set up hot tail grid
ds.hottailgrid.setEnabled(False)

# Disable runaway grid
ds.runawaygrid.setEnabled(False)

# Set solver type and parameters
ds.solver.setType(DREAM.Solver.LINEAR_IMPLICIT)

# Save settings to HDF5 file
ds.save('self_consistent_fluid_dream_settings.h5')

# Run simulation
DREAM.runiface(ds)

# Restart simulation twice
for i in range(2):
    ds = DREAMSettings('self_consistent_fluid_dream_settings.h5')
    ds.timestep.setNt(2000)
    ds.save('self_consistent_fluid_dream_settings_restart{}.h5'.format(i+1))
    DREAM.runiface(ds)