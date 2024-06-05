 from DREAM.Settings.DriftDiffusion import *
from DREAM.Settings.Equations import *
from DREAM.Settings.InitialProfiles import *
from DREAM.Settings.OhmicHeating import *
from DREAM.Settings.Radiation import *
from DREAM.Settings.RunawayElectrons import *
from DREAM.Settings.SpitzerResistivity import *
from DREAM.Data.Output import *
from DREAM import PlasmaState

# Set up radial grid
nrad = 50
rg = RadialGrid(0, 1, nrad, 1.0e-3)
addSetting(rg)

# Set time steps
nt = 100
addSetting(nt)
addSetting(0.0, 'finalTime')

# Add ions
ionList = ['D', 'D3+', 'D+']
addIon(ionList)

# Set up temperature and electric field
addSetting(Temperature(rg, 10.0))
addSetting(ElectricField(rg, 0.0))

# Disable runaway and hot-tail grid
addSetting(False, 'runawayOn')
addSetting(False, 'hotTailOn')

# Initialize simulation
runSimulation()

# Ionization
setIonization(Ionization(rg, 'D', 'D+', 1.0e-12, 10.0, 10.0, 0.0))
runSimulation()

# Equilibration
changeRadialGrid(rg)
runSimulation()

# Radiation
setRadiation(Radiation(rg, 'D', 'D+', 1.0e-12, 10.0, 10.0, 0.0))
runSimulation()

# Save settings
s = PlasmaState(getTimeAndUnit(), rg, getTemperature(), ElectricField(rg, 0.0))
saveSettings(s)

# Plot energy balance
plotEnergyBalance()