 ```python
import DREAM.Settings.Equations.Fluid as Fluid
import DREAM.Settings.Equations.Kinetic as Kinetic
import DREAM.Settings.Grid as Grid
import DREAM.Settings.DriftKinetic as DriftKinetic
import DREAM.Settings.TimeStepping as TimeStepping
import DREAM.Settings.General as General
import DREAM.Plotting as Plotting
import h5py

# Initialize settings
settings = General.DREAMSettings()

# Set up the simulation parameters
settings.equations = Fluid.FluidEquations()
settings.drift_kinetic_equations = Kinetic.KineticEquations()
settings.drift_kinetic_equations.number_of_passing_electrons = 0
settings.drift_kinetic_equations.number_of_trapped_electrons = 0
settings.drift_kinetic_equations.number_of_ions = 1
settings.drift_kinetic_equations.ion_species = ["D+"]

# Set up the radial grid
settings.grid = Grid.RadialGrid()
settings.grid.number_of_zones = 50
settings.grid.inner_radius = 0
settings.grid.outer_radius = 1

# Set the time stepper
settings.time_stepping = TimeStepping.TimeStepping()
settings.time_stepping.number_of_time_steps = 100
settings.time_stepping.time_step_length = 1e-7

# Add ions
settings.drift_kinetic_equations.add_ion("D+", 3.5e6, 1.5)

# Set the electric field and cold electron temperature
settings.electric_field = DriftKinetic.BoltzmannElectronDistribution(1.0)
settings.cold_electron_temperature = DriftKinetic.ConstantColdElectronTemperature(10.0)

# Set up the hot tail grid
settings.hot_tail_grid = DriftKinetic.HotTailGrid()
settings.hot_tail_grid.number_of_zones = 5
settings.hot_tail_grid.minimum_energy = 10.0
settings.hot_tail_grid.maximum_energy = 100.0

# Disable the runaway grid
settings.runaway_grid = DriftKinetic.RunawayGrid()
settings.runaway_grid.number_of_zones = 0

# Set the solver type and its parameters
settings.solver = DriftKinetic.IterativeSolver()
settings.solver.iterative_method = DriftKinetic.IterativeMethod.PIC
settings.solver.preconditioner = DriftKinetic.Preconditioner.DIAGONAL
settings.solver.number_of_pic_iterations = 5
settings.solver.convergence_criterion = 1e-5

# Save settings to HDF5 file
with h5py.File('run1.h5', 'w') as f:
    settings.save(f)

# Run the simulation
DREAM.run(settings)

# Restart the simulation
with h5py.File('run1.h5', 'r') as f:
    settings = General.DREAMSettings(f)

settings.time_stepping.number_of_time_steps = 200
settings.time_stepping.time_step_length = 5e-8

with h5py.File('run2.h5', 'w') as f:
    settings.save(f)

DREAM.run(settings)

# Restart the simulation again
with h5py.File('run2.h5', 'r') as f:
    settings = General.DREAMSettings(f)

settings.time_stepping.number_of_time_steps = 300
settings.time_stepping.time_step_length = 1e-7

with h5py.File('run3.h5', 'w') as f:
    settings.save(f)

DREAM.run(settings)
```