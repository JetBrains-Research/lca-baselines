```python
import numpy as np
import porepy as pp

def add_data(gb, domain):
    for g, d in gb:
        # Assign permeability
        perm = pp.SecondOrderTensor(np.ones(g.num_cells))
        d[pp.PARAMETERS] = {"second_order_tensor": perm}
        
        # Assign apertures for fractures
        if g.dim < gb.dim_max():
            aperture = np.power(0.1, gb.dim_max() - g.dim)
            d[pp.PARAMETERS]["aperture"] = np.ones(g.num_cells) * aperture
        
        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(["dir"] * bc_faces.size)
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_values = np.zeros(g.num_faces)
        bc_values[bc_faces] = 1.0
        d[pp.PARAMETERS]["bc"] = bc
        d[pp.PARAMETERS]["bc_values"] = bc_values

def plot_over_line(gb, start_point, end_point, num_points, variable, filename):
    line = np.linspace(start_point, end_point, num_points)
    values = pp.plot_over_line(gb, line, variable)
    np.savetxt(filename, values, delimiter=",")

tol = 1e-5
mesh_size = {"mesh_size_frac": 0.1, "mesh_size_min": 0.02, "mesh_size_bound": 0.1}
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

file_path = "path_to_csv_file.csv"
gb = pp.fracs.simplex.create_grid_bucket_2d_from_csv(file_path)
pp.contact_conditions.set_projections(gb)
gb.compute_geometry()
gb.coarsen()
gb.assign_node_ordering()

add_data(gb, domain)

solver = pp.DualVEMMixDim(gb)
A, b = solver.matrix_rhs()
solution = solver.solve(A, b)

split_solution = pp.fvutils.split_dofs(gb, solution)
pressure = split_solution[0]
discharge = split_solution[1]
pp.fvutils.project_discharge(gb, discharge, "vector_source")

pp.io.grid_bucket_to_vtk(gb, "grid_bucket_vtk", data_names=["pressure", "vector_source"])

bbox = np.array([[0, 1], [0, 1]])
num_points = 100
x_line = np.vstack((np.linspace(bbox[0, 0], bbox[0, 1], num_points), np.ones(num_points) * np.mean(bbox[1, :])))
y_line = np.vstack((np.ones(num_points) * np.mean(bbox[0, :]), np.linspace(bbox[1, 0], bbox[1, 1], num_points)))

plot_over_line(gb, x_line[:, 0], x_line[:, -1], num_points, "pressure", "pressure_x_line.csv")
plot_over_line(gb, y_line[:, 0], y_line[:, -1], num_points, "pressure", "pressure_y_line.csv")

diam = gb.diameter()
num_cells_2d = sum(g.num_cells for g, d in gb if g.dim == 2)
num_cells_1d = sum(g.num_cells for g, d in gb if g.dim == 1)
print(f"Diameter of the grid bucket: {diam}")
print(f"Number of cells in 2D: {num_cells_2d}, in 1D: {num_cells_1d}")
```