 Here is the Python code based on your instructions using the porepy library:

```python
import numpy as np
from porepy import DarcyAndTransport, GridBucket, GmshGridBucketWriter, SnapToGridTest

tolerance = 1e-6

def add_data(gb, bucket_id, permeability, apertures, boundary_conditions):
    gb.set_property(bucket_id, 'permeability', permeability)
    gb.set_property(bucket_id, 'apertures', apertures)
    gb.set_property(bucket_id, 'boundary_conditions', boundary_conditions)

def plot_over_line(gb, line, variable):
    # Implement the plotting logic here
    pass

# Set mesh size parameters and domain
mesh_size = 0.1
domain = [(0, 1), (0, 1)]

# Import grid bucket from CSV file, compute its geometry, coarsen it, and assign node ordering
gb = GridBucket.from_gmsh('path/to/your/mesh.msh')
gb.compute_geometry()
gb.coarsen(mesh_size)
gb.assign_node_ordering()

# Use the add_data function to assign parameters to the grid bucket
add_data(gb, 0, np.ones(gb.dof_of_grid()), np.ones(gb.dof_of_grid()), 'dirichlet')

# Define a solver
solver = DarcyAndTransport(gb, tolerance=tolerance)
A, b = solver.compute_matrix_and_rhs()
solution = np.linalg.solve(A, b)

# Split the solution, extract the discharge and pressure, and project the discharge
discharge, pressure = solver.split_solution(solution)
pressure = solver.project_discharge(discharge)

# Export the grid bucket to a VTK file, including the pressure and discharge
GmshGridBucketWriter(gb).write('output.vtk', pressure, discharge)

# Define a bounding box and a number of points
bounding_box = [(0, 0), (1, 1)]
num_points = 100

# Define two sets of points along the x and y axes
x_points = np.linspace(bounding_box[0][0], bounding_box[0][1], num_points)
y_points = np.linspace(bounding_box[1][0], bounding_box[1][1], num_points)
x_points, y_points = np.meshgrid(x_points, y_points)

# Use the plot_over_line function to plot the pressure along these lines and save the results to CSV files
for i in range(num_points):
    for j in range(num_points):
        line = [(x_points[i, j], y_points[i, j]), (x_points[i, j+1], y_points[i, j+1])]
        pressure_values = plot_over_line(gb, line, pressure)
        np.savetxt(f'pressure_line_{i}_{j}.csv', pressure_values, delimiter=',')

# Print the diameter of the grid bucket and the number of cells in 2D and 1D
print(f'Diameter of the grid bucket: {gb.diameter}')
print(f'Number of cells in 2D: {gb.num_cells_2d}')
print(f'Number of cells in 1D: {gb.num_cells_1d}')
```

Please note that you need to implement the `plot_over_line` function according to your specific requirements. Also, replace `'path/to/your/mesh.msh'` with the path to your Gmsh mesh file.

This code assumes that you have the porepy library installed. If not, you can install it using pip:

```
pip install porepy
```