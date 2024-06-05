```python
import numpy as np
import porepy as pp

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        # Permeability
        perm = pp.SecondOrderTensor(np.ones(g.num_cells))  # Isotropic permeability
        d[pp.PARAMETERS] = pp.Parameters(g)
        d[pp.PARAMETERS]["second_order_tensor"] = perm
        
        # Source term
        source_term = np.zeros(g.num_cells)
        d[pp.PARAMETERS]["source"] = source_term
        
        # Aperture
        aperture = np.ones(g.num_cells)
        d[pp.PARAMETERS]["aperture"] = aperture
        
        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(["dir"] * bc_faces.size)
        bc_values = np.zeros(g.num_faces)
        bc_values[bc_faces] = 1.0  # Example values
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        d[pp.PARAMETERS]["bc"] = bc
        d[pp.PARAMETERS]["bc_values"] = bc_values

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        # Source term
        source_term = np.zeros(g.num_cells)
        d[pp.PARAMETERS]["source"] = source_term
        
        # Porosity
        porosity = np.ones(g.num_cells)
        d[pp.PARAMETERS]["porosity"] = porosity
        
        # Discharge
        discharge = np.zeros(g.num_cells)
        d[pp.PARAMETERS]["discharge"] = discharge
        
        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(["neu"] * bc_faces.size)
        bc_values = np.zeros(g.num_faces)
        bc_values[bc_faces] = 0.0  # Example values
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        d[pp.PARAMETERS]["bc"] = bc
        d[pp.PARAMETERS]["bc_values"] = bc_values

# Variables
tol = 1e-5
export_folder = "results"
time = 0
num_time_steps = 10
time_step_size = 1.0
export_frequency = 2
coarsen = False
mesh_size = {"mesh_size_frac": 0.1, "mesh_size_min": 0.02}
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

# Import grid
csv_file = "path_to_grid.csv"
grid_bucket = pp.import_grid_from_csv(csv_file, mesh_size, domain, tol)

# Compute geometry and coarsen if necessary
grid_bucket.compute_geometry()
if coarsen:
    pp.coarsen_grid(grid_bucket)

# Assign node ordering
pp.assign_node_ordering(grid_bucket)

# Create Darcy solver and add data
darcy_solver = pp.Tpfa("flow")
add_data_darcy(grid_bucket, domain, tol)

# Solve Darcy problem
darcy_solver.discretize(grid_bucket)
darcy_solver.solve(grid_bucket)

# Extract and project discharge and pressure
pp.project_discharge_and_pressure(grid_bucket)

# Compute total flow rate and export results
total_flow_rate = pp.compute_total_flow(grid_bucket)
pp.export_vtk(grid_bucket, file_name=f"{export_folder}/darcy_solution", time_step=0)

# Define variables for physics
physics = "advection"

# Create advection and mass matrix solvers
advection_solver = pp.Upwind(physics)
mass_matrix_solver = pp.MassMatrix(physics)

# Add advection data
add_data_advection(grid_bucket, domain, tol)

# Add time step property
for g, d in grid_bucket:
    d[pp.PARAMETERS][physics]["time_step"] = time_step_size

# Create matrices and right-hand sides
advection_solver.discretize(grid_bucket)
mass_matrix_solver.discretize(grid_bucket)

# LU factorization
advection_solver.prepare_solver(grid_bucket)
mass_matrix_solver.prepare_solver(grid_bucket)

# Initialize solution vector and arrays for time and production
solution = np.zeros(grid_bucket.num_cells())
times = np.zeros(num_time_steps)
production = np.zeros(num_time_steps)

# Time loop
for step in range(num_time_steps):
    time += time_step_size
    # Update solution
    advection_solver.solve(grid_bucket)
    mass_matrix_solver.solve(grid_bucket)
    
    # Compute production
    production[step] = np.sum(solution)  # Example calculation
    
    # Export solution
    if step % export_frequency == 0:
        pp.export_vtk(grid_bucket, file_name=f"{export_folder}/advection_solution", time_step=step)
    
    times[step] = time

# Export time steps to PVD file
pp.export_pvd(grid_bucket, file_name=f"{export_folder}/time_steps", times=times)

# Save times and production to text file
np.savetxt(f"{export_folder}/times_and_production.txt", np.vstack((times, production)).T)
```