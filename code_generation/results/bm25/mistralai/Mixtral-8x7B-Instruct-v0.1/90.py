 ```python
import DREAM.Settings.DREAMSettings as DREAMSettings
import DREAM.InitialProfiles as InitialProfiles
import DREAM.Equations.EquationBase as EquationBase
import DREAM.RadialGrid as RadialGrid
import DREAM.MagneticField as MagneticField
import DREAM.Fluid as Fluid
import DREAM.Output as Output
import DREAM.Evolution as Evolution
import DREAM.Util.units as units
import numpy as np

# Create DREAM settings
settings = DREAMSettings.DREAMSettings()

# Set up simulation parameters
settings.simulation.Nt = 1000
settings.simulation.tEnd = 1e-3
settings.simulation.tWrite = 1e-5
settings.simulation.tSave = 1e-5
settings.profiles.Nr = 100
settings.profiles.boundaryConditionType = 0

# Set up radial grid
settings.grids.radialGrid.type = 'AlcatorCMod'
settings.grids.radialGrid.R0 = 0.68
settings.grids.radialGrid.a = 0.22
settings.grids.radialGrid.Rmin = 0.5
settings.grids.radialGrid.Rmax = 0.85

# Set time stepper
settings.timeIntegration.type = 'Explicit'

# Add ions
settings.species.add(
    name='D', Z=1, A=2, massDensity0=0.3e-4, temperature0=100, resistivityModel='Braginskii'
)
settings.species.add(
    name='T', Z=1, A=3, massDensity0=0.0, temperature0=1000, resistivityModel='Braginskii'
)

# Set electric field and cold electron temperature
settings.electromagnetics.E0 = ElectricField.constantE(0.0)
settings.species.D.temperature0 = 100

# Set up hot tail grid
settings.hotTail.enable = True
settings.hotTail.Nr = 100
settings.hotTail.Rmin = 0.5
settings.hotTail.Rmax = 0.85
settings.hotTail.dR = 0.001
settings.hotTail.Tmin = 100
settings.hotTail.Tmax = 1000
settings.hotTail.dT = 10

# Disable runaway grid
settings.runaway.enable = False

# Set solver type and its parameters
settings.solver.type = 'Gear'
settings.solver.Gear.order = 2
settings.solver.Gear.stabilityLimit = 0.5

# Include other necessary settings
settings.evolution.fluid = Fluid.Fluid(
    EquationBase.IonisationEquation,
    EquationBase.ContinuityEquation,
    EquationBase.MomentumEquation,
    EquationBase.EnergyEquation,
)
settings.evolution.fluidTE = Fluid.Fluid(
    EquationBase.IonisationEquation,
    EquationBase.ContinuityEquation,
    EquationBase.MomentumEquation,
    EquationBase.EnergyEquation,
    EquationBase.EnergyEquation,
)
settings.evolution.fluidTZ = Fluid.Fluid(
    EquationBase.IonisationEquation,
    EquationBase.ContinuityEquation,
    EquationBase.MomentumEquation,
    EquationBase.EnergyEquation,
    EquationBase.EnergyEquation,
)
settings.evolution.fluidNE = Fluid.Fluid(
    EquationBase.IonisationEquation,
    EquationBase.ContinuityEquation,
    EquationBase.MomentumEquation,
    EquationBase.EnergyEquation,
)

# Save settings to HDF5 file
Output.saveSettings(settings, 'run1.h5')

# Run the simulation
runSimulation('run1.h5')

# Restart the simulation
for i in range(2):
    # Load settings from output of previous run
    settings = loadCODE('run' + str(i) + '.h5')

    # Adjust time stepper
    settings.timeIntegration.type = 'Implicit'

    # Save new settings to HDF5 file
    Output.saveSettings(settings, 'run' + str(i + 1) + '.h5')

    # Run the simulation
    runSimulation('run' + str(i + 1) + '.h5')
```