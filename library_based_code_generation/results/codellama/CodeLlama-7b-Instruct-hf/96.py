  import numpy as np
import pandas as pd
from poropy import GridBucket, Domain, DarcySolver, AdvectionSolver, MassMatrixSolver
from poropy.utils import coarsen_grid
from poropy.io import read_grid_csv
from poropy.solvers import LU
from poropy.utils import project_to_nodes
from poropy.io import write_vtk, write_pvd, write_text

# Define functions for adding data to grid bucket
def add_data_darcy(gb, tol):
    gb.add_data('permeability', np.array([[1, 0], [0, 1]]), tol)
    gb.add_data('source', np.array([[0, 0], [0, 0]]), tol)
    gb.add_data('aperture', np.array([[0, 0], [0, 0]]), tol)
    gb.add_data('boundary_conditions', np.array([[0, 0], [0, 0]]), tol)

def add_data_advection(gb, tol):
    gb.add_data('source', np.array([[0, 0], [0, 0]]), tol)
    gb.add_data('porosity', np.array([[1, 0], [0, 1]]), tol)
    gb.add_data('discharge', np.array([[0, 0], [0, 0]]), tol)
    gb.add_data('boundary_conditions', np.array([[0, 0], [0, 0]]), tol)

# Define variables for tolerance, export folder, time, number of time steps, time step size, export frequency, and coarsening
tol = 1e-3
export_folder = 'output'
time = 0
num_time_steps = 100
time_step_size = 0.01
export_frequency = 10
coarsen = True

# Define dictionaries for mesh size and domain boundaries
mesh_size = {'x': 10, 'y': 10}
domain_boundaries = {'left': 0, 'right': 1, 'bottom': 0, 'top': 1}

# Import grid from CSV file, compute geometry, coarsen if necessary, and assign node ordering
grid = read_grid_csv('grid.csv')
grid = coarsen_grid(grid, mesh_size) if coarsen else grid
grid.assign_node_ordering()

# Create Darcy solver, add Darcy data to grid bucket, solve Darcy problem, and extract and project discharge and pressure
darcy_solver = DarcySolver(grid)
gb = GridBucket(grid)
add_data_darcy(gb, tol)
darcy_solver.solve(gb)
discharge = project_to_nodes(gb.get_data('discharge'), grid)
pressure = project_to_nodes(gb.get_data('pressure'), grid)

# Compute total flow rate and export results to VTK file
total_flow_rate = np.sum(discharge)
write_vtk(grid, discharge, pressure, f'{export_folder}/discharge.vtk')

# Define variables for physics, create advection and mass matrix solvers, add advection data to grid bucket, and add time step property to grid bucket
physics = 'advection'
advection_solver = AdvectionSolver(grid, physics)
mass_matrix_solver = MassMatrixSolver(grid, physics)
gb = GridBucket(grid)
add_data_advection(gb, tol)
gb.add_property('time_step', time_step_size)

# Create matrices and right-hand sides for advection and mass matrix problems, perform LU factorization to speed up solver
A_advection = advection_solver.assemble_matrix()
b_advection = advection_solver.assemble_rhs()
A_mass_matrix = mass_matrix_solver.assemble_matrix()
b_mass_matrix = mass_matrix_solver.assemble_rhs()
lu = LU(A_advection)

# Initialize solution vector and arrays for time and production, loop over time steps to update solution, compute production, and export solution to VTK file every specified number of time steps
solution = np.zeros(gb.num_nodes)
time_array = np.zeros(num_time_steps)
production_array = np.zeros(num_time_steps)
for i in range(num_time_steps):
    time += time_step_size
    gb.set_property('time_step', time_step_size)
    solution = lu.solve(b_advection)
    production = np.sum(solution)
    time_array[i] = time
    production_array[i] = production
    if i % export_frequency == 0:
        write_vtk(grid, solution, f'{export_folder}/production_{i}.vtk')

# Export time steps to PVD file and save times and absolute production values to text file
write_pvd(f'{export_folder}/production.pvd', time_array, production_array)
write_text(f'{export_folder}/production.txt', time_array, production_array)