 Here is a basic example of how you might set up a radial transport simulation using the DREAM library. Please note that this is a simplified example and you may need to adjust it according to your specific needs. Also, I'm assuming you have DREAM installed and the necessary imports are already done.

```python
from dream.physics import *
from dream.grid import *
from dream.ions import *
from dream.electrons import *
from dream.transport import *

n_re = 1.0e19  # initial number density of ions (cm^-3)
T_i_init = 1.0e6  # initial ion temperature (K)
T_e_init = 1.0e4  # initial electron temperature (K)
T_i_final = 2.0e6  # final ion temperature (K)
T_e_final = 1.0e4  # final electron temperature (K)
D_ion = 1.0e10  # diffusion coefficient (cm^2/s)
n_grid = 100  # number of grid points
r_min = 1.0  # inner radius (cm)
r_max = 10.0  # outer radius (cm)
dt = 1.0e-9  # time step (s)
n_steps = 1000  # number of time steps
E_field = 1.0e3  # electric field (V/cm)

grid = RadialGrid(n_grid, r_min, r_max)
ions = IonSpecies(n_re, 'O+', mass=16.0, charge=8.0)
electrons = ColdElectrons(T_e_init)

ion_transport = IonTransport(ions, electrons, D_ion)
electron_transport = ElectronTransport(electrons)

ion_transport.set_E_field(E_field)
ion_transport.set_hot_tail_grid(True)

sim = Simulation(grid, [ion_transport, electron_transport])
sim.set_boundary_conditions(BoundaryConditions.fixed_flux)

for step in range(n_steps):
    sim.run_step(dt)
    if step % 100 == 0:
        print(f"Step {step}: T_i = {sim.get_ion_temperature(0):.2f} K")

T_i_final_sim = sim.get_ion_temperature(0)
print(f"Final ion temperature: T_i = {T_i_final_sim:.2f} K")
```

This code sets up a radial transport simulation with a single ion species (O+) and cold electrons. The ion diffusion coefficient is set to a constant value, and the electric field and cold electron temperature are also set. The hot tail grid is enabled, and the simulation runs for a specified number of time steps. The ion temperature is printed every 100 steps, and the final ion temperature is printed at the end.