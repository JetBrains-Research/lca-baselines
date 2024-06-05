 Here is the Python code that follows your instructions, using the porepy library:

```python
import porepy as pp
import numpy as np
import vtk

def add_data_to_bucket(bucket, permeability, apertures, boundary_conditions, coupling_permeability):
    for grid in bucket:
        grid.set_property('k', permeability)
        grid.set_property('apertures', apertures)
        grid.set_boundary_conditions(boundary_conditions)
        for edge in grid.edges():
            edge.set_property('k', coupling_permeability)

def write_network_to_csv(network, filename):
    with open(filename, 'w') as f:
        for point in network:
            f.write(f"{point[0]}, {point[1]}\n")

def main(permeability_factor, description, mesh_size):
    # Load network from CSV file
    network = pp.read_network_from_points(read_points_from_file('network.csv'))

    # Create grid bucket
    bucket = pp.GridBucket(network, mesh_size=mesh_size)

    # Compute geometry
    bucket.compute_geometry()

    # Optionally generate a coarse grid
    bucket.generate_coarse_grid()

    # Assign parameters
    add_data_to_bucket(bucket, permeability_factor, [0.01], {}, 0.0)

    # Solve system of equations
    solver = pp.solver.DualVEMMixDim()
    solution = solver.solve(bucket)

    # Extract and project the solution
    pressure = solution.get_pressure()
    saturation = solution.get_saturation()

    # Export results to VTK file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(pressure.to_vtk())
    writer.SetFileName(f"results_{description}.vtk")
    writer.Write()

    # Print out some information about the grid bucket
    print(f"Grid bucket with {len(bucket)} grids, mesh size {mesh_size}")

def read_points_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    points = [list(map(float, line.split(','))) for line in lines]
    return points

def blocking_scenario():
    main(permeability_factor=1.0, description="blocking", mesh_size=0.01)

def permeable_scenario():
    main(permeability_factor=0.1, description="permeable", mesh_size=0.05)

# Write network to CSV file
write_network_to_csv(pp.read_points_from_file('network.csv'), 'network.csv')

# Call blocking and permeable scenarios
blocking_scenario()
permeable_scenario()
```

This code assumes that you have a CSV file named 'network.csv' containing the network points. You may need to adjust the network file and the function `read_points_from_file` according to your specific network format. Also, make sure you have the porepy library installed. If not, you can install it using pip:

```
pip install porepy
```