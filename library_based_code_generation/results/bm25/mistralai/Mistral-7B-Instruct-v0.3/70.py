 Here is the Python code based on your instructions:

```python
import numpy as np
import scipy.sparse as sp
from porepy import GridBucket, DarcyAndTransport, GmshGridBucketWriter, TestGettersAndSetters

def add_data(grid_bucket, domain, permeability_factor):
    grid_bucket.permeability = permeability_factor * np.ones(domain.shape)
    grid_bucket.source_term = 0.0
    grid_bucket.apertures = np.ones(domain.shape)
    grid_bucket.boundary_conditions = {'left': 0.0, 'right': 0.0, 'bottom': 0.0, 'top': 0.0}

def write_network(filename, network_string):
    with open(filename, 'w') as file:
        file.write(network_string)

def main(permeability_factor, description, coarsen, export_results):
    # Define mesh parameters
    mesh_size = 100
    domain = np.zeros((mesh_size, mesh_size))

    # Create a grid bucket
    grid_bucket = GridBucket(domain)

    # Write a network to a file
    network_string = "your_network_string_here"
    write_network("network.msh", network_string)

    # Import a 2D fracture network from the file
    grid_bucket.import_gmsh("network.msh")

    # Compute the geometry of the grid bucket
    grid_bucket.compute_geometry()

    if coarsen:
        grid_bucket.coarsen()

    # Assign an ordering to the nodes of the grid bucket
    grid_bucket.order_nodes()

    # Add data to the grid bucket
    add_data(grid_bucket, domain, permeability_factor)

    # Define solvers for flow and source
    flow_solver = DarcyAndTransport(grid_bucket)
    source_solver = TestGettersAndSetters(grid_bucket)

    # Compute the right-hand side and the matrix of the linear systems
    rhs, A = flow_solver.assemble_in_bucket()

    # Solve the linear systems
    b = sp.linalg.solve(A, rhs)

    # Split the solution
    discharge, pressure = flow_solver.split_solution(b)

    # Extract the discharge and pressure from the solution
    discharge = discharge.reshape(domain.shape)
    pressure = pressure.reshape(domain.shape)

    # Project the discharge
    discharge = flow_solver.project_discharge(discharge)

    if export_results:
        # Export the pressure and the projected discharge to a vtk file
        pressure_writer = GmshGridBucketWriter(grid_bucket)
        pressure_writer.write_vtk("pressure.vtk", pressure)
        discharge_writer = GmshGridBucketWriter(grid_bucket)
        discharge_writer.write_vtk("discharge.vtk", discharge)

def test_vem_blocking():
    main(0.001, "blocking", False, False)

def test_vem_permeable():
    main(10.0, "permeable", False, False)
```

Please note that you need to replace `"your_network_string_here"` with the actual network string you want to use. Also, the code assumes that you have the necessary network string and the porepy library installed. If you don't have the porepy library installed, you can install it using pip:

```bash
pip install porepy
```