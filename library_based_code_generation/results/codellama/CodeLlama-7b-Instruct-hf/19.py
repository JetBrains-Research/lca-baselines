  from dream import *

# Set up the simulation parameters
sim = Simulation(
    name='Runaway Scenario',
    grid=Grid(
        momentum_max=1e6,
        momentum_grid_points=300,
        pitch_grid_points=20,
        time_steps=20,
        time_max=1e-3,
    ),
    solver=Solver(
        type='backward_euler',
    ),
    time_stepper=TimeStepper(
        type='rk4',
    ),
    output=Output(
        file='output.h5',
    ),
)

# Set up the physical parameters
sim.set_physical_parameters(
    electric_field_strength=6,
    electron_density=5e19,
    temperature=100,
)

# Set up the radial grid
sim.set_radial_grid(
    grid_points=100,
    grid_spacing=0.1,
)

# Run the simulation
sim.run()