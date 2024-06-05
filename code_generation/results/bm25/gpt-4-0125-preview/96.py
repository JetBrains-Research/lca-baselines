```python
import numpy as np
import porepy as pp

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        # Permeability
        perm = pp.SecondOrderTensor(np.ones(g.num_cells))  # Isotropic permeability
        # Source term
        source_term = np.zeros(g.num_cells)
        # Aperture
        aperture = np.ones(g.num_cells)
        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(['dir'] * bc_faces.size)
        bc_values = np.zeros(g.num_faces)
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        
        specified_parameters = {"second_order_tensor": perm, "source": source_term,
                                "bc": bc, "bc_values": bc_values, "aperture": aperture}
        pp.initialize_data(g, d, "flow", specified_parameters)

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        # Source term
        source_term = np.zeros(g.num_cells)
        # Porosity
        porosity = np.ones(g.num_cells)
        # Discharge
        discharge = np.zeros(g.num_faces)
        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(['neu'] * bc_faces.size)
        bc_values = np.zeros(g.num_faces)
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        
        specified_parameters = {"source": source_term, "bc": bc, "bc_values": bc_values,
                                "mass_weight": porosity, "darcy_flux": discharge}
        pp.initialize_data(g, d, "transport", specified_parameters)

# Variables
tol = 1e-5
export_folder = "results"
time = 0
num_time_steps = 100
time_step_size = 0.1
export_frequency = 10
coarsen = False
mesh_size = {"mesh_size_frac": 0.1, "mesh_size_min": 0.02}
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

# Import grid
csv_file = "path_to_grid.csv"
grid_bucket = pp.import_grid_from_csv(csv_file, domain=domain, tol=tol)

# Compute geometry and coarsen if necessary
grid_bucket.compute_geometry()
if coarsen:
    pp.coarsen_grid(grid_bucket)

# Assign node ordering
pp.assign_node_ordering(grid_bucket)

# Create Darcy solver and add data
darcy_solver = pp.DarcyAndTransport()
add_data_darcy(grid_bucket, domain, tol)

# Solve Darcy problem
darcy_solver.discretize(gb=grid_bucket)
darcy_solver.solve(gb=grid_bucket)

# Extract and project discharge and pressure
pressure = grid_bucket.get_node_prop("pressure")
discharge = grid_bucket.get_node_prop("darcy_flux")

# Compute total flow rate and export results
total_flow_rate = np.sum(discharge)
pp.to_vtk(grid_bucket, file_name=export_folder+"/darcy_solution", data={"pressure": pressure, "discharge": discharge})

# Define variables for physics
physics = "advection"

# Create advection and mass matrix solvers
advection_solver = pp.Upwind()
mass_matrix_solver = pp.MassMatrix()

# Add advection data
add_data_advection(grid_bucket, domain, tol)

# Add time step property
grid_bucket.add_node_prop("time_step", time_step_size)

# Create matrices and RHS for advection and mass matrix problems
advection_solver.discretize(gb=grid_bucket)
mass_matrix_solver.discretize(gb=grid_bucket)

# LU factorization
advection_solver.factorize(gb=grid_bucket)
mass_matrix_solver.factorize(gb=grid_bucket)

# Initialize solution vector and arrays for time and production
solution = np.zeros(grid_bucket.num_cells())
times = np.zeros(num_time_steps)
production = np.zeros(num_time_steps)

# Time loop
for step in range(num_time_steps):
    time += time_step_size
    # Update solution
    solution = advection_solver.solve(gb=grid_bucket, previous_solution=solution)
    # Compute production
    production[step] = np.sum(solution)
    # Export solution
    if step % export_frequency == 0:
        pp.to_vtk(grid_bucket, file_name=f"{export_folder}/solution_{step}", data={"solution": solution})
    times[step] = time

# Export time steps to PVD file
pp.to_pvd(export_folder, times, file_name="time_steps.pvd")

# Save times and absolute production values
np.savetxt(f"{export_folder}/times_and_production.txt", np.vstack((times, np.abs(production))).T)
```