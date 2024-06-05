```python
import DREAM

# Set up radial grid
rGrid = DREAM.RadialGrid(0, 1, 100)

# Set time steps
tMax = 1e-3
times = DREAM.tBeforeOnsetFromQCritAndTcoldFromOutput(rGrid, tMax)

# Add ions
DREAM.addSetting('eqsys', 'n_i', 'DREAM.Settings.Equations.Ionization.QUICK_IONIZATION_MODELS')

# Set up temperature and electric field
T = DREAM.Temperature(rGrid)
E = DREAM.ElectricField(rGrid)

# Disable runaway and hot-tail grid
DREAM.addSetting('eqsys', 'hotTailGrid', False)
DREAM.addSetting('eqsys', 'runawayGrid', False)

# Initialization stage
DREAM.run(times[0])

# Ionization stage
DREAM.setIonization(rGrid, times[1])
DREAM.run(times[1])

# Equilibration stage
DREAM.run(times[2])

# Radiation stage
DREAM.run(times[3])

# Save settings and run simulation
DREAM.runSimulation()

# Plot results
DREAM.plotEnergyBalance()
```