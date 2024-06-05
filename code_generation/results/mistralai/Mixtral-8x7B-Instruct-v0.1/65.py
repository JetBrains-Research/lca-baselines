 ```python
import dream

# Set up simulation parameters
sim = dream.DreamSim()
sim.n_e = 1e19
sim.T_e_i = 100
sim.T_e_f = 200
sim.time_steps = 100
sim.species = ['D', 'T']
sim.E_field = 0.1
sim.T_cold_e = 10
sim.hot_tail_grid = True

# Set up radial grid
r_min, r_max = 0.1, 1.0
n_r = 50
sim.radial_grid = dream.RadialGrid(r_min, r_max, n_r)

# Set up time stepper
sim.time_stepper = dream.TimeStepper(sim.time_steps)

# Set up ions
sim.ions = dream.Ions(sim.species)

# Set E_field and cold electron temperature
sim.set_E_field(sim.E_field)
sim.set_T_cold_e(sim.T_cold_e)

# Enable hot tail grid
sim.enable_hot_tail_grid()

# Set up transport settings
if sim.hot_tail_grid:
    sim.transport_settings = dream.TransportSettings(diffusion_coefficient=1.0)
else:
    if sim.T_e_i == sim.T_e_f:
        sim.transport_settings = dream.TransportSettings(diffusion_coefficient=0.0)
    else:
        sim.transport_settings = dream.TransportSettings(diffusion_coefficient=1.0)

# Run simulation
sim.run()
```