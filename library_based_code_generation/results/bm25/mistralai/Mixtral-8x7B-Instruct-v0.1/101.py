 ```python
import porepy as pp
from porepy.grids.grid_generators import gmsh_grid
from porepy.numerics.solvers import DualVEMMixDim
import numpy as np
import vtk
import os


def add_data(grid_bucket, domain):
    k = pp.models.darcy.models.constant_permeability(grid_bucket, 1.0)
    a = pp.models.darcy.models.constant_aperture(grid_bucket, 0.1)
    bc = pp.models.darcy.models.dirichlet_bc(grid_bucket, 1.0, domain)
    return k, a, bc


def plot_over_line(grid_bucket, var, line_points):
    line_cells = grid_bucket.edge_cells(grid_bucket.find_edge(line_points))
    line_values = var[line_cells]
    return line_values


tol = 1e-12
h = 0.1
domain = pp.Domain(dim=2, comm=pp.worldcomm)
gb = gmsh_grid('grid.msh', domain)
gb.compute_geometry()
gb.coarsen(tol=tol, max_volume=h**2)
gb.set_node_ordering()

k, a, bc = add_data(gb, domain)

solver = DualVEMMixDim(gb, k=k, a=a, bc=bc)
system = solver.create_system()
system.assemble()
rhs = system.create_rhs()
rhs.set_constant(0)
sol = system.solve()

pressure, discharge = sol.split()
discharge = pp.postprocess.project_to_grid(discharge, gb)

gb.write_vtk('gb.vts', point_data={'pressure': pressure, 'discharge': discharge})

box = [(0, 0), (1, 1)]
n_points = 10
x_points = np.linspace(*box[0], n_points)
y_points = np.linspace(*box[1], n_points)

line_points_x = np.array([(x, 0.5) for x in x_points])
line_points_y = np.array([(0.5, y) for y in y_points])

pressure_x = plot_over_line(gb, pressure, line_points_x)
pressure_y = plot_over_line(gb, pressure, line_points_y)

np.savetxt('pressure_x.csv', pressure_x, delimiter=',')
np.savetxt('pressure_y.csv', pressure_y, delimiter=',')

print(f'Diameter: {gb.diameter()}')
print(f'Number of 2D cells: {gb.num_2d_cells()}')
print(f'Number of 1D cells: {gb.num_1d_cells()}')
```