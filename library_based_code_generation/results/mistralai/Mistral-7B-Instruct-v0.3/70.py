 Here is the Python code based on your instructions. Please note that this code assumes that you have already installed the necessary modules (numpy, scipy, and porepy). Also, the predefined network string and the structure of the functions might need adjustments according to your specific requirements.

```python
import numpy as np
import scipy.sparse as sp
from porepy import Mesh, GridBucket, FlowSolver, SourceSolver, VTKExporter

def add_data(grid_bucket, domain, permeability_factor):
    grid_bucket.set_property('permeability', permeability_factor)
    grid_bucket.set_property('source_term', np.ones(domain.volume()))
    grid_bucket.set_property('apertures', np.ones(domain.surface()))
    grid_bucket.set_boundary_condition('fixed_pressure', 0, domain.boundary())

def write_network(filename, network_string):
    with open(filename, 'w') as f:
        f.write(network_string)

def main(permeability_factor, description, coarsen, export_results):
    mesh = Mesh.create_rectangular(100, 100, 1)
    domain = mesh.domain()
    filename = f"{description}_network.pnt"
    write_network(filename, "predefined_network_string")

    network = Mesh.read_network(filename)
    grid_bucket = GridBucket(mesh, network)
    grid_bucket.refine(2)

    if coarsen:
        grid_bucket.coarsen()

    ordering = grid_bucket.get_node_ordering()
    grid_bucket.set_property('ordering', ordering)

    add_data(grid_bucket, domain, permeability_factor)

    flow_solver = FlowSolver(grid_bucket)
    source_solver = SourceSolver(grid_bucket)

    A = flow_solver.discrete_form()
    b = source_solver.discrete_source()

    solver = sp.linalg.cg
    tol = 1e-12
    max_iter = 1000

    x = solver(A, b, tol=tol, maxiter=max_iter)

    u = x[ordering.inverse()]
    p = u[:grid_bucket.num_nodes()]
    q = u[grid_bucket.num_nodes():]

    if export_results:
        exporter = VTKExporter(mesh)
        exporter.export_pressure(p, f"{description}_pressure.vtk")
        exporter.export_discharge(q, f"{description}_discharge.vtk")

def test_vem_blocking():
    main(0.01, "blocking", False, False)

def test_vem_permeable():
    main(1000, "permeable", False, True)
```

This code defines the four functions as requested, and the test functions call the main function with the specified parameters. The main function creates a mesh, reads a predefined network, refines the grid bucket, optionally coarsens it, assigns an ordering to the nodes, adds data to the grid bucket, solves the linear systems, and exports the results if necessary.