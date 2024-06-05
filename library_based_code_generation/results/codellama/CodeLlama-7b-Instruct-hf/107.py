  import porepy as pp

def add_data_to_grid_bucket(grid_bucket, permeability, apertures, boundary_conditions, coupling_permeability):
    for grid in grid_bucket:
        grid.permeability = permeability
        grid.apertures = apertures
        grid.boundary_conditions = boundary_conditions
        grid.coupling_permeability = coupling_permeability

def write_network_to_csv(network, file_name):
    with open(file_name, 'w') as f:
        f.write(network)

def main(permeability_factor, description, mesh_size):
    # Create a grid bucket from a CSV file
    grid_bucket = pp.GridBucket.from_csv('grid_bucket.csv')

    # Compute the geometry of the grid bucket
    grid_bucket.compute_geometry()

    # Optionally generate a coarse grid
    # grid_bucket.generate_coarse_grid(mesh_size)

    # Assign parameters to the grid bucket
    grid_bucket.permeability_factor = permeability_factor
    grid_bucket.description = description

    # Solve a system of equations using the DualVEMMixDim solver
    solver = pp.DualVEMMixDim(grid_bucket)
    solver.solve()

    # Extract and project the solution
    solution = solver.extract_solution()
    projected_solution = solver.project_solution(solution)

    # Export the results to a VTK file
    pp.export_to_vtk(grid_bucket, 'results.vtk')

    # Print out some information about the grid bucket
    print(f'Grid bucket: {grid_bucket}')
    print(f'Permeability factor: {permeability_factor}')
    print(f'Description: {description}')
    print(f'Mesh size: {mesh_size}')
    print(f'Solution: {solution}')
    print(f'Projected solution: {projected_solution}')

def blocking_scenario():
    main(0.5, 'Blocking scenario', 10)

def permeable_scenario():
    main(1.0, 'Permeable scenario', 20)

blocking_scenario()
permeable_scenario()