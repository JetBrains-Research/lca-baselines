 ```python
import os
import sys
import numpy as np
from scipy.sparse import linalg
import porepy as pp

sys.path.append('path/to/soultz_grid')
import soultz_grid

tol = 1e-10
gb = pp.GridBucket(tol=tol)
domain = pp.Domain(grid=gb)

def add_data_darcy(gb, tol):
    """Add Darcy parameters to a grid bucket."""
    param = pp.Darcy()
    param['permeability'] = pp.Constant('permeability', 1e-12)
    param['source'] = pp.Constant('source', 0.0)
    param['aperture'] = pp.Constant('aperture', 1e-4)
    param['boundary_conditions'] = pp.BoundaryCondition(
        'dirichlet', 'pressure', 1.0, apply_to_faces=True)
    pp.add_disconnected_components(gb, param, tol)

def add_data_advection(gb, tol):
    """Add advection parameters to a grid bucket."""
    param = pp.Advection()
    param['source'] = pp.Constant('source', 0.0)
    param['porosity'] = pp.Constant('porosity', 1.0)
    param['discharge'] = pp.Constant('discharge', 0.0)
    param['boundary_conditions'] = pp.BoundaryCondition(
        'dirichlet', 'concentration', 0.0, apply_to_faces=True)
    pp.add_disconnected_components(gb, param, tol)

# Set up parameters for creating a grid
grid_params = soultz_grid.GridParameters()
grid_params.tol = tol

# Create a grid
grid = soultz_grid.create_grid(grid_params)

# Compute geometry
pp.compute_geometry(grid)

# Coarsen grid if necessary
if grid.num_cells > 1e6:
    grid = pp.coarsen_grid(grid, 'max_cell_volume', tol=tol)

# Assign node ordering
grid = pp.assign_node_ordering(grid)

# Solve Darcy problem
darcy_solver = pp.DualVEMMixDim()
darcy_prob = pp.DarcyProblem(grid, darcy_solver)
darcy_prob.solve()

# Add Darcy parameters to grid bucket
add_data_darcy(gb, tol)

# Compute matrix and right-hand side
A, b = pp.compute_system_matrix_rhs(gb, 'darcy')

# Solve system of equations
sol = linalg.spsolve(A, b)

# Split solution
pressure, discharge = pp.split_solution(sol, gb)

# Project discharge
discharge = pp.project_to_faces(discharge, gb)

# Compute total flow rate
flow_rate = discharge.dot(grid.face_areas)

# Set up parameters for transport problem
transport_params = pp.TransportParameters()

# Define solvers for advection and mass matrix
adv_solver = pp.AdvectionSolver()
mass_solver = pp.MassMatrixSolver()

# Add transport parameters to grid bucket
add_data_advection(gb, tol)

# Compute matrices and right-hand side vectors
A_adv, b_adv = pp.compute_system_matrix_rhs(gb, 'advection', solver=adv_solver)
A_mass, b_mass = pp.compute_system_matrix_rhs(gb, 'mass_matrix', solver=mass_solver)

# Perform LU factorization
A_adv_lu = linalg.splu(A_adv)

# Initialize solution vector
sol_vec = np.zeros(gb.num_vars)

# Time-stepping loop
for t in np.arange(0, 10, 0.1):
    # Update source term
    gb['source']['advection'].data[:] = np.exp(-t)

    # Perform matrix-vector multiplication
    rhs = A_adv_lu * sol_vec

    # Update solution vector
    sol_vec[:] = pp.solve_linear(A_mass, rhs - b_mass)

    # Export solution at certain time steps
    if np.mod(t, 1) == 0:
        pp.exporter.PVD(gb, 'transport_solution.pvd', time_step=t)

# Export solution in PVD format
pp.exporter.PVD(gb, 'transport_solution.pvd')

# Save production data to text file
np.savetxt('production_data.txt', flow_rate)
```