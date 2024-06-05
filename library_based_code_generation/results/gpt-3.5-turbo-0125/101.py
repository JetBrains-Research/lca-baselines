import porepy as pp

def add_data(grid_bucket, domain):
    # Define permeability, apertures, and boundary conditions
    pass

def plot_over_line(grid_bucket, values):
    # Plot values over a line in a grid bucket
    pass

tolerance = 1e-6
mesh_size = 0.1
domain = pp.Box([0, 0], [1, 1])

grid_bucket = pp.GridBucket()
grid_bucket.create_grid(domain, mesh_size)
grid_bucket.compute_geometry()
grid_bucket.coarsen()
grid_bucket.assign_node_ordering()

add_data(grid_bucket, domain)

solver = pp.DualVEMMixDim(grid_bucket)
solver.compute_geometry()
solver.assemble_matrix_rhs()
solver.solve()

solution = solver.split()
discharge, pressure = solution.extract_discharge_pressure()
projected_discharge = solution.project_discharge()

pp.export_to_vtk(grid_bucket, "output.vtk", {"pressure": pressure, "discharge": discharge})

bounding_box = pp.Box([0, 0], [1, 1])
num_points = 10
points_x = pp.create_points_along_line(bounding_box, num_points, direction="x")
points_y = pp.create_points_along_line(bounding_box, num_points, direction="y")

plot_over_line(grid_bucket, pressure)
plot_over_line(grid_bucket, discharge)

print("Diameter of grid bucket:", grid_bucket.diameter())
print("Number of cells in 2D:", grid_bucket.num_cells())
print("Number of cells in 1D:", grid_bucket.num_cells(1))