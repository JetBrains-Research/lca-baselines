 ```python
import porepy as pp
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

def add_data(grid_bucket, domain):
    k = pp.constants.ConstantProperty(grid_bucket, 1e-12)
    a = pp.constants.ConstantProperty(grid_bucket, 1e-5)
    bcs = pp.bc.DirichletBC(grid_bucket, 1, domain.boundaries['inlet'])
    return k, a, bcs

tolerance = 1e-8
h = 1.0
domain = pp.Domain(dim=2, max_cells=1000)

grid_bucket = pp.import_mesh(Path('grid.csv'), domain)
grid_bucket.compute_geometry()
grid_bucket.coarsen(tolerance=tolerance, method='max_volume')
grid_bucket.set_order(pp.Ordering.cell_centers)

k, a, bcs = add_data(grid_bucket, domain)
solver = pp.DualVEMMixDim(grid_bucket, k, a, bcs)
A, b = solver.matrix_and_rhs()
solution = solver.solve(A, b)

pressure, discharge = pp.postprocess.split_solution(solution, grid_bucket)
projected_discharge = pp.postprocess.project_discharge(grid_bucket, discharge)

pp.export_vtk(grid_bucket, Path('output.vtk'), pressure, projected_discharge)

bbox = grid_bucket.bounding_box()
n_points = 10
x_points = np.linspace(bbox[0], bbox[1], n_points)
y_points = np.linspace(bbox[2], bbox[3], n_points)

pressure_x = pp.postprocess.plot_over_line(grid_bucket, pressure, x_points, 'x')
pressure_y = pp.postprocess.plot_over_line(grid_bucket, pressure, y_points, 'y')

np.savetxt('pressure_x.csv', pressure_x, delimiter=',')
np.savetxt('pressure_y.csv', pressure_y, delimiter=',')

diameter = grid_bucket.diameter()
n_2d_cells = grid_bucket.num_cells_dim(2)
n_1d_cells = grid_bucket.num_cells_dim(1)

print(f'Diameter: {diameter:.2f}')
print(f'Number of 2D cells: {n_2d_cells}')
print(f'Number of 1D cells: {n_1d_cells}')
```