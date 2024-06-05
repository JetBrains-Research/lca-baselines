import numpy as np
import scipy
import os
import sys
from porepy import *

def add_data_darcy(gb, tol):
    # Add parameters related to Darcy's law
    gb.add_node_props(["permeability", "source", "aperture", "bc"])

def add_data_advection(gb, tol):
    # Add parameters related to advection
    gb.add_node_props(["source", "porosity", "discharge", "bc"])

sys.path.append('path/to/module')
import soultz_grid

# Set up parameters for creating a grid
grid_params = {"nx": 10, "ny": 10, "nz": 10}
grid = soultz_grid.create_grid(grid_params)

# Compute geometry, coarsen grid, assign node ordering
grid.compute_geometry()
grid.coarsen()
grid.assign_node_ordering()

# Solve Darcy problem
solver = DualVEMMixDim()
gb.add_node_props(["permeability", "source", "aperture", "bc"])
A, b = solver.assemble_matrix_rhs(gb)
sol = solver.solve(A, b)
pressure, discharge = solver.split(sol)
discharge_projected = solver.project_discharge(discharge)
total_flow_rate = solver.compute_total_flow_rate(discharge_projected)

# Set up parameters for transport problem
advection_solver = AdvectionSolver()
mass_matrix_solver = MassMatrixSolver()
gb.add_node_props(["source", "porosity", "discharge", "bc"])
A_adv, b_adv = advection_solver.assemble_matrix_rhs(gb)
A_mass, b_mass = mass_matrix_solver.assemble_matrix_rhs(gb)
lu = scipy.sparse.linalg.splu(A_adv)
sol = np.zeros_like(b_adv)

# Time-stepping loop
for t in range(num_time_steps):
    sol = lu.solve(b_adv - A_mass @ sol)
    if t % output_interval == 0:
        export_solution(sol, t)

# Export solution in PVD format
export_solution_pvd(sol)

# Save production data to text file
save_production_data_to_text_file()