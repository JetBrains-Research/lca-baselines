 Here is the Python code based on your instructions using the porepy library:

```python
import porepy as pp
import numpy as np

def add_data_to_bucket(bucket, permeability, apertures, boundary_conditions, coupling_permeability):
    bucket.set_permeability(permeability)
    bucket.set_apertures(apertures)
    bucket.set_boundary_conditions(boundary_conditions)
    bucket.set_coupling_permeability(coupling_permeability)

def write_network_to_csv(network, filename):
    with open(filename, 'w') as f:
        for point in network:
            f.write(f"{point[0]}, {point[1]}\n")

def main(permeability_factor, description, mesh_size):
    points = pp.utils.read_points_from_file('network.csv')
    bucket = pp.GridBucket.from_points(points, mesh_size)
    bucket.snap_to_grid()
    add_data_to_bucket(bucket, permeability_factor, apertures=None, boundary_conditions=None, coupling_permeability=None)
    solver = pp.DarcyAndTransport.DualVEMMixDim()
    solver.solve(bucket)
    solution = solver.get_solution()
    bucket.project_solution(solution)
    bucket.to_vtk(f"{description}_result.vtk")
    print(f"Grid bucket {description} solved.")

def blocking_scenario():
    main(permeability_factor=1e-12, description="blocking", mesh_size=0.01)

def permeable_scenario():
    main(permeability_factor=1e-8, description="permeable", mesh_size=0.1)

blocking_scenario()
permeable_scenario()
```

Please note that this code assumes you have a CSV file named 'network.csv' containing a list of points in the format:

```
x1 y1
x2 y2
...
```

You may need to adjust the permeability factor, mesh size, and network file according to your specific needs. Also, you may need to install the porepy library using pip:

```
pip install porepy
```