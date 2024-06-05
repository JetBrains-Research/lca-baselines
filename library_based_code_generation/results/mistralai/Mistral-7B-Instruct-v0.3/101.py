 ```python
import numpy as np
import porepy as pp
from porepy.utils import read_csv_bucket

def add_data(bucket, domain, permeability, apertures, bc):
    bucket.set_property('k', permeability)
    bucket.set_property('aperture', apertures)
    bucket.set_boundary_conditions(bc)

def plot_over_line(bucket, line, property_name):
    x, y = line
    values = bucket.get_property(property_name, x, y)
    return values

tolerance = 1e-8
mesh_size = (1, 1)
domain = pp.Domain(tolerance=tolerance)

bucket = read_csv_bucket(domain, 'bucket.csv')
bucket.compute_geometry()
bucket.coarsen(factor=2)
bucket.assign_node_ordering()

add_data(bucket, domain, permeability=1e-14, apertures=1e-6, bc={'left': 0, 'right': 0, 'bottom': 0, 'top': 0})

solver = pp.solvers.DualVEMMixDim(domain)
A, b = solver.compute_system(bucket)
solution = solver.solve(A, b)

discharge, pressure = solution.split()
discharge = discharge.project()

bucket.export_vtk('bucket.vtk', pressure=pressure, discharge=discharge)

bbox = (0, 1, 0, 1)
n_points = 100
x_points = np.linspace(bbox[0], bbox[2], n_points)
y_points = np.linspace(bbox[1], bbox[3], n_points)

lines = [(x, y) for x in x_points for y in y_points]

with open('pressure_x.csv', 'w') as f:
    for line in lines:
        pressure_val = plot_over_line(bucket, line, 'pressure')
        f.write(f'{line}, {pressure_val}\n')

with open('pressure_y.csv', 'w') as f:
    for line in lines:
        pressure_val = plot_over_line(bucket, line, 'pressure')
        f.write(f'{line}, {pressure_val}\n')

print(f'Diameter: {bucket.diameter}')
print(f'Number of 2D cells: {bucket.num_cells}')
print(f'Number of 1D cells: {len(bucket.nodes)}')
```