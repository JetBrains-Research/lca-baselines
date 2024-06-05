 ```python
import os
import sys
import numpy as np
from scipy.sparse import linalg
import porepy as pp

# Append the porepy directory to the system path
sys.path.append(os.path.dirname(pp.__file__))

from porepy.grid.soultz_grid import SoultzGrid
from porepy.flow.darcy import DualVEMMixDim
from porepy.transport.advection import AdvectionSolver
from porepy.transport.mass_matrix import MassMatrix
from porepy.utils.boundary_conditions import DirichletBC, NeumannBC

def add_data_darcy(gb, tol, permeability, source, aperture, bc):
    gb.set_property('permeability', permeability)
    gb.set_property('source', source)
    gb.set_property('aperture', aperture)
    gb.set_boundary_condition(bc)

def add_data_advection(gb, tol, porosity, discharge, bc):
    gb.set_property('porosity', porosity)
    gb.set_property('discharge', discharge)
    gb.set_boundary_condition(bc)

# Set up parameters for creating a grid
grid_params = {
    'n_cells': (100, 100, 100),
    'cell_type': 'hexahedron',
    'min_size': 0.01,
    'max_size': 0.1,
    'coarsen_factor': 1.2,
    'node_ordering': 'natural'
}

# Create a grid
grid = SoultzGrid(**grid_params)
grid.compute_geometry()
if grid.max_size() > grid_params['max_size']:
    grid.coarsen(grid_params['coarsen_factor'])
grid.assign_node_ordering()

# Solve a Darcy problem
solver = DualVEMMixDim(grid, tol=tol)
solver.add_data(add_data_darcy)
A, b = solver.compute_system()
u = linalg.solve(A, b)
u_split = np.split(u, 3, axis=1)
p, q = u_split[0], u_split[1]
q_projected = pp.discretizations.project(q, grid)
flow_rate = np.sum(q_projected)

# Set up parameters for a transport problem
transport_params = {
    'dt': 0.01,
    'n_steps': 1000,
    'output_steps': 100
}

# Define solvers for advection and mass matrix
advection_solver = AdvectionSolver(grid, tol=tol)
mass_matrix = MassMatrix(grid)

# Add data to the grid bucket for the transport problem
add_data_advection(grid.gb, tol, porosity=1, discharge=np.zeros(grid.n_nodes), bc=DirichletBC(grid, 1))

# Compute matrices and right-hand side vectors
M, f = mass_matrix.compute_system()
A_adv, b_adv = advection_solver.compute_system()

# Perform an LU factorization
L, U = linalg.lu(M)

# Initialize a solution vector
u = np.zeros(grid.n_nodes)

# Perform a time-stepping loop to update the solution and export it at certain time steps
for i in range(transport_params['n_steps']):
    if i % transport_params['output_steps'] == 0:
        pp.io.write_pvd(grid, u, f'transport_step_{i}.pvd')
    u_new = u + transport_params['dt'] * advection_solver(u)
    u = linalg.solve_triangular(L, linalg.solve_triangular(L, u_new - f, trans=True) - U * u_new)

# Export the solution in PVD format
pp.io.write_pvd(grid, u, 'final_solution.pvd')

# Save the production data to a text file
with open('production_data.txt', 'w') as f:
    f.write(f'Total flow rate: {flow_rate}\n')
```