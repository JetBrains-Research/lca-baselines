 Here is the Python code for setting up a simple runaway scenario simulation using the DREAM library:

```python
from dream.grid import MomentumGrid, RadialGrid
from dream.physics import ElectricField, RunawayElectronDensity
from dream.solver import TimeStepper
from dream.ion import Hydrogen
from dream.units import format_time_and_unit, get_time_and_unit
from dream.util import _generateMomentumGrid, set_custom_grid_points, set_biuniform_grid, set_custom_grid
from dream.output import HDF5Output

# Physical parameters
electric_field = ElectricField(strength=6 * 1e-3)
density = 5e19 * 1e-3
temperature = 100 * 1.602e-19

# Grid parameters
max_momentum = 1 * 9.109e-31 * 3e8
num_momentum_points = 300
num_pitch_points = 20
time_step = 1e-6
num_time_steps = 20

# Radial grid
radial_grid = RadialGrid(min_radius=0, max_radius=100, num_points=100)

# Momentum grid
momentum_grid = _generateMomentumGrid(max_momentum, num_momentum_points)
set_biuniform_grid(momentum_grid)

# Set up the grid
grid = Grid(radial_grid, momentum_grid)

# Set up ions
ion = Hydrogen()
ion_index = ionNameToIndex(ion.name)

# Set up runaway electron density
runaway_electron_density = RunawayElectronDensity(density, temperature)

# Set up solver type
solver_type = "drift_kinetic"

# Set up time stepper
time_stepper = TimeStepper(solver_type, grid, runaway_electron_density, ion, time_step)

# Set up output
output = HDF5Output("output.h5")
set_custom_grid_points(output.radial_grid, radial_grid.points)
set_custom_grid_points(output.momentum_grid, momentum_grid.points)
set_custom_grid(output, grid)

# Initialize to equilibrium
initialize_to_equilibrium(time_stepper, output)

# Run simulation
run_simulation(time_stepper, output, num_time_steps)

# Set number of save steps
set_number_of_save_steps(output, num_time_steps)

# Change radial grid (if needed)
# changeRadialGrid(time_stepper, radial_grid)

# Print time and unit information
print(f"Time and unit: {format_time_and_unit(get_time_and_unit(time_step))}")
```

Please note that this code assumes you have the DREAM library installed and properly configured. Also, the commented lines are for changing the radial grid if needed. You can uncomment those lines and adjust the radial grid as per your requirements.