  Here is the Python code that sets up a simple runaway scenario simulation using the DREAM library:
```
from dream import *

# Set up the simulation parameters
electric_field_strength = 6.0
electron_density = 5e19
temperature = 100.0

# Set up the grid parameters
max_momentum = 1.0 * me * c
num_momentum_grid_points = 300
num_pitch_grid_points = 20
simulation_time = 1e-3
num_time_steps = 20

# Set up the radial grid
radial_grid = RadialGrid(num_pitch_grid_points, max_momentum)

# Set up the solver type
solver_type = 'cg'

# Set up the time stepper
time_stepper = TimeStepper(simulation_time, num_time_steps)

# Set up the electric field
electric_field = ElectricField(electric_field_strength)

# Set up the runaway electron density
runaway_electron_density = RunawayElectronDensity(electron_density)

# Set up the simulation
simulation = Simulation(radial_grid, solver_type, time_stepper, electric_field, runaway_electron_density)

# Set up the output
output = Output('output.h5')

# Run the simulation
simulation.run()

# Save the output
output.save()
```
This code sets up a simple runaway scenario simulation using the DREAM library, with the specified physical parameters and grid parameters. The simulation is run for the specified time, and the output is saved to an HDF5 file named 'output.h5'.