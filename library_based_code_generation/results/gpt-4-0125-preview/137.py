```python
import numpy as np
import scipy.sparse as sps
import os
import sys
import porepy as pp

# Append path and import custom module
sys.path.append('/path/to/directory')
import soultz_grid

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        # Permeability
        perm = pp.SecondOrderTensor(np.ones(g.num_cells)) 
        d[pp.PARAMETERS] = pp.Parameters(g)
        d[pp.PARAMETERS]["second_order_tensor"] = perm
        
        # Source term
        d[pp.PARAMETERS]["source"] = np.zeros(g.num_cells)
        
        # Aperture
        if g.dim < gb.dim_max():
            d[pp.PARAMETERS]["aperture"] = np.ones(g.num_cells) * 1e-4
        
        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(["dir"] * bc_faces.size)
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_values = np.zeros(g.num_faces)
        d[pp.PARAMETERS]["bc"] = bc
        d[pp.PARAMETERS]["bc_values"] = bc_values

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        # Source term
        d[pp.PARAMETERS]["source"] = np.zeros(g.num_cells)
        
        # Porosity
        d[pp.PARAMETERS]["porosity"] = np.ones(g.num_cells)
        
        # Discharge
        d[pp.PARAMETERS]["discharge"] = np.zeros(g.num_faces)
        
        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(["neu"] * bc_faces.size)
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_values = np.zeros(g.num_faces)
        d[pp.PARAMETERS]["bc"] = bc
        d[pp.PARAMETERS]["bc_values"] = bc_values

# Set up parameters for creating a grid
params = {"mesh_size_frac": 0.02, "mesh_size_min": 0.01, "mesh_size_bound": 0.1}
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

# Create grid
gb = soultz_grid.create_grid(params, domain)

# Compute geometry and coarsen if needed
gb.compute_geometry()
if gb.dim_max() > 2:
    pp.coarsen_grid(gb)

# Assign node ordering
pp.contact_conditions.set_projections(gb)

# Solve Darcy problem
tol = 1e-5
add_data_darcy(gb, domain, tol)
solver = pp.DualVEMMixDim(gb)
A, b = solver.matrix_rhs()
solution = sps.linalg.spsolve(A, b)
split_solution = solver.split(solution)
discharge, pressure = split_solution[0], split_solution[1]
projected_discharge = solver.project_discharge(discharge)
total_flow_rate = np.abs(projected_discharge).sum()

# Set up parameters for transport problem
time_step = 1e-3
num_steps = 100
export_every = 10

# Define solvers for advection and mass matrix
advection_solver = pp.Upwind()
mass_matrix_solver = pp.MassMatrix()

# Add parameters for advection
add_data_advection(gb, domain, tol)

# Compute matrices and right-hand side vectors
advection_matrix, advection_rhs = advection_solver.discretize(gb)
mass_matrix, _ = mass_matrix_solver.discretize(gb)

# Perform LU factorization
lu = sps.linalg.splu(mass_matrix)

# Initialize solution vector
solution = np.zeros(advection_rhs.shape)

# Time-stepping loop
for step in range(num_steps):
    solution = lu.solve(solution + time_step * (advection_matrix * solution + advection_rhs))
    if step % export_every == 0:
        pp.exporter.export_vtk(gb, file_name="solution_" + str(step), data={"solution": solution})

# Export solution in PVD format
pp.exporter.export_vtk(gb, "final_solution", data={"solution": solution})

# Save production data
np.savetxt("production_data.txt", total_flow_rate)
```