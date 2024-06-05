  import numpy as np
import scipy.sparse as sp
import porepy as pp

def add_data(grid_bucket, domain, permeability_factor):
    # Add parameters to the grid bucket
    grid_bucket.add_parameter("permeability", permeability_factor)
    grid_bucket.add_parameter("source_term", 1.0)
    grid_bucket.add_parameter("apertures", np.zeros((domain.n_nodes, 2)))
    grid_bucket.add_parameter("boundary_conditions", np.zeros((domain.n_nodes, 2)))

def write_network(file_name):
    # Write a predefined network string to a file with the given name
    with open(file_name, "w") as f:
        f.write("network\n")
        f.write("1 2\n")
        f.write("2 3\n")
        f.write("3 4\n")
        f.write("4 1\n")
        f.write("end\n")

def main(permeability_factor, description, coarsen_grid_bucket, export_results):
    # Define mesh parameters, domain, and file name
    mesh_parameters = pp.MeshParameters(1.0, 1.0, 10, 10)
    domain = pp.Domain(mesh_parameters)
    file_name = "network.txt"

    # Write a network to a file
    write_network(file_name)

    # Import a 2D fracture network from the file
    network = pp.Network.from_file(file_name)

    # Compute the geometry of the grid bucket
    grid_bucket = pp.GridBucket.from_network(network, domain)

    # Optionally coarsen the grid bucket
    if coarsen_grid_bucket:
        grid_bucket = grid_bucket.coarsen()

    # Assign an ordering to the nodes of the grid bucket
    grid_bucket.assign_ordering()

    # Add data to the grid bucket
    add_data(grid_bucket, domain, permeability_factor)

    # Define solvers for flow and source
    flow_solver = pp.FlowSolver(grid_bucket)
    source_solver = pp.SourceSolver(grid_bucket)

    # Compute the right-hand side and the matrix of the linear systems
    rhs = flow_solver.compute_rhs() + source_solver.compute_rhs()
    matrix = flow_solver.compute_matrix() + source_solver.compute_matrix()

    # Solve the linear systems
    solution = sp.linalg.spsolve(matrix, rhs)

    # Split the solution
    flow_solution = solution[:grid_bucket.n_nodes]
    source_solution = solution[grid_bucket.n_nodes:]

    # Extract the discharge and pressure from the solution
    discharge = flow_solution * grid_bucket.permeability
    pressure = flow_solution * grid_bucket.permeability

    # Project the discharge
    projected_discharge = pp.project(discharge, grid_bucket)

    # Export the pressure and the projected discharge to a vtk file
    if export_results:
        pp.export_vtk(pressure, "pressure.vtk")
        pp.export_vtk(projected_discharge, "projected_discharge.vtk")

def test_vem_blocking():
    main(0.1, "blocking", False, False)

def test_vem_permeable():
    main(10.0, "permeable", False, False)