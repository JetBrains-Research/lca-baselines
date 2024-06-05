 import dream
from dream. Const import Const
from dream. Plasma import Plasma
from dream. Runaway import Runaway
from dream. Solvers import ImplicitMoment
from dream. TimeSteppers import RungeKutta4
from dream. RadialGrids import UniformRadialGrid
from dream. Outputs import HDF5Output

# Physical parameters
E = 6  # V/m
ne = 5e19  # 1/m^3
T = 100  # eV

# Grid parameters
max_p = Const.m_e * Const.c  # m_e*c
n_p = 300
n_pitch = 20
dt = 1e-3  # s
n_timesteps = 20

# Set up the simulation
plasma = Plasma(ne=ne, T=T, Z=1)
runaway = Runaway(plasma, Efield=E)
runaway.set_momentum_grid(max_p, n_p)
runaway.set_pitch_grid(n_pitch)
runaway.set_radial_grid(UniformRadialGrid(n_radial=1))
runaway.set_solver(ImplicitMoment())
runaway.set_time_stepper(RungeKutta4(dt))

# Set up the output
output = HDF5Output('output.h5')
output.add_runaway(runaway)
output.write_header()

# Run the simulation
for i in range(n_timesteps):
    runaway.solve(n_steps=1)
    output.write_timestep(i)

output.close()