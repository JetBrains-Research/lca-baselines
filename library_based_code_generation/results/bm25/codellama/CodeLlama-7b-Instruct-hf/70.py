  Here is the code you requested:
```
import numpy as np
import scipy.sparse as sp
import porepy as pp

def add_data(grid_bucket, domain, permeability_factor):
    # Add parameters to the grid bucket
    grid_bucket.permeability = permeability_factor
    grid_bucket.source_term = np.zeros(grid_bucket.num_nodes)
    grid_bucket.apertures = np.zeros(grid_bucket.num_nodes)
    grid_bucket.boundary_conditions = np.zeros(grid_bucket.num_nodes)

def write_network(file_name):
    # Write a predefined network string to a file with the given name
    with open(file_name, 'w') as f:
        f.write('network\n')
        f.write('nodes\n')
        f.write('edges\n')
        f.write('end\n')

def main(permeability_factor, description, coarsen_grid_bucket, export_results):
    # Define mesh parameters, domain, and file name
    mesh_parameters = {'element_type': 'triangle', 'num_elements': 10}
    domain = pp.Domain(mesh_parameters)
    file_name = 'network.msh'

    # Write a network to a file
    write_network(file_name)

    # Import a 2D fracture network from the file
    network = pp.Network.from_file(file_name)

    # Compute the geometry of the grid bucket
    grid_bucket = pp.GridBucket(network, domain)

    # Optionally coarsen the grid bucket
    if coarsen_grid_bucket:
        grid_bucket.coarsen()

    # Assign an ordering to the nodes of the grid bucket
    grid_bucket.assign_ordering()

    # Add data to the grid bucket
    add_data(grid_bucket, domain, permeability_factor)

    # Define solvers for flow and source
    flow_solver = pp.DarcyAndTransport(grid_bucket, domain)
    source_solver = pp.DarcyAndTransport(grid_bucket, domain)

    # Compute the right-hand side and the matrix of the linear systems
    rhs = flow_solver.compute_rhs()
    matrix = flow_solver.compute_matrix()

    # Solve the linear systems
    solution = sp.linalg.solve(matrix, rhs)

    # Split the solution
    flow_solution = solution[:grid_bucket.num_nodes]
    source_solution = solution[grid_bucket.num_nodes:]

    # Extract the discharge and pressure from the solution
    discharge = flow_solution[:grid_bucket.num_edges]
    pressure = flow_solution[grid_bucket.num_edges:]

    # Project the discharge
    projected_discharge = pp.project(discharge, grid_bucket, domain)

    # Export the pressure and the projected discharge to a vtk file
    if export_results:
        pp.export_vtk(pressure, projected_discharge, file_name)

def test_vem_blocking():
    main(0.1, 'blocking', False, False)

def test_vem_permeable():
    main(100, 'permeable', False, False)
```