```python
import numpy as np
import scipy.sparse as sp
import os
import sys

import porepy as pp

# Append path and import custom module
sys.path.append('/path/to/soultz_grid')
import soultz_grid

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        # Permeability
        perm = pp.SecondOrderTensor(np.ones(g.num_cells))
        d[pp.PARAMETERS] = pp.Parameters(g, ["flow"], {"second_order_tensor": perm})
        
        # Source term
        f = np.zeros(g.num_cells)
        d[pp.PARAMETERS]["flow"]["source"] = f
        
        # Aperture
        aperture = np.ones(g.num_cells)
        d[pp.PARAMETERS]["flow"]["aperture"] = aperture
        
        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(["dir"] * bc_faces.size)
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_values = np.zeros(g.num_faces)
        d[pp.PARAMETERS]["flow"]["bc"] = bc
        d[pp.PARAMETERS]["flow"]["bc_values"] = bc_values

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        # Source term
        f = np.zeros(g.num_cells)
        d[pp.PARAMETERS] = pp.Parameters(g, ["transport"], {"source": f})
        
        # Porosity
        porosity = np.ones(g.num_cells)
        d[pp.PARAMETERS]["transport"]["porosity"] = porosity
        
        # Discharge
        discharge = np.zeros(g.num_faces)
        d[pp.PARAMETERS]["transport"]["discharge"] = discharge
        
        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(["neu"] * bc_faces.size)
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_values = np.zeros(g.num_faces)
        d[pp.PARAMETERS]["transport"]["bc"] = bc
        d[pp.PARAMETERS]["transport"]["bc_values"] = bc_values

# Grid parameters
params = {"mesh_size_frac": 0.02, "mesh_size_min": 0.01, "mesh_size_bound": 0.1}
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

# Create grid
gb = soultz_grid.create_grid(params)

# Compute geometry and coarsen
gb.compute_geometry()
if gb.dim_max() > 2:
    pp.coarsening.coarsen(gb, method="by_volume", coarsen_factor=0.5)
gb.assign_node_ordering()

# Solve Darcy problem
solver = pp.DualVEMMixDim(params)
solver.discretize(gb)
A, b = solver.assemble_matrix_rhs(gb)
solution = sp.linalg.spsolve(A, b)
split_solution = solver.distribute_variable(gb, solution)
discharge, pressure = solver.extract_discharge_pressure(gb, split_solution)
solver.project_discharge(gb, discharge)
total_flow = solver.compute_total_flow(gb)

# Transport problem setup
time_step = 1e-4
end_time = 1e-2
current_time = 0
solver_advection = pp.Upwind(params)
solver_mass = pp.MassMatrix(params)

# Add parameters for transport
add_data_advection(gb, domain, tol=1e-5)

# Compute matrices and rhs for transport
A_transport, b_transport = solver_advection.assemble_matrix_rhs(gb)
M, _ = solver_mass.assemble_matrix_rhs(gb)

# LU factorization
lu = sp.linalg.splu(A_transport)

# Initialize solution vector
solution_transport = np.zeros(b_transport.shape)

# Time-stepping loop
while current_time < end_time:
    current_time += time_step
    b_transport = M * solution_transport
    solution_transport = lu.solve(b_transport)
    
    # Export solution at certain time steps
    if current_time % (end_time / 10) == 0:
        pp.exporter.to_vtk(gb, "solution_transport", {"concentration": solution_transport})

# Export final solution in PVD format
pp.exporter.to_vtk(gb, "solution_transport_final", {"concentration": solution_transport})

# Save production data
np.savetxt("production_data.txt", solution_transport)
```