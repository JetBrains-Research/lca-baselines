 Here is the Python code for setting up a simple runaway scenario simulation using the DREAM library:

```python
from dream.physics import *
from dream.grid import *
from dream.solver import *
from dream.io import *

# Physical parameters
E_field = 6 * volt / meter
n_e = 5e19 * meter**-3
T_e = 100 * electron_volt

# Grid parameters
p_max = electron_mass * speed_of_light
num_p_points = 300
num_r_points = 20
dt = 1e-3 / num_r_points

# Radial grid
r_grid = log_spaced_grid(1e-3, 100 * meter, num_r_points)

# Momentum grid
p_grid = lin_spaced_grid(0, p_max, num_p_points)

# Solver and time stepper
solver = PIC(n_e, T_e, E_field)
time_stepper = RungeKutta4(dt)

# Set up the simulation
sim = Simulation(solver, time_stepper, r_grid, p_grid)

# Apply electric field
sim.apply_electric_field(E_field, direction='x')

# Run the simulation
sim.run(1e-3)

# Save the output to an HDF5 file
HDF5Writer(sim, 'output.h5')
```

This code sets up a simple runaway scenario simulation using the DREAM library with the specified physical and grid parameters. The output is saved to an HDF5 file named 'output.h5'.