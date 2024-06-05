 ```python
import porepy as pp
import numpy as np
import csv
from vedo import Volume, Plotter, show

def add_data_to_grid_bucket(grid_bucket):
    for grid in grid_bucket:
        # Permeability
        grid.props['permeability'] = np.ones((grid.num_cells,)) * 1e-12

        # Apertures
        grid.props['apertures'] = np.ones((grid.num_faces,)) * 1e-5

        # Boundary conditions
        bc_true = np.ones((grid.num_faces,), dtype=bool)
        grid.props['boundary_conditions'] = (bc_true, bc_true)

        # Coupling permeability
        for edge in grid.edges:
            if edge.grid_ids[0] != edge.grid_ids[1]:
                edge.props['permeability'] = 1e-14

def write_network_to_csv(network, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(network)

def main(permeability_factor, description, mesh_size, generate_coarse_grid=False):
    # Create grid bucket from CSV file
    gb = pp.grid_bucket.GridBucket.from_file(f'mesh_{mesh_size}.csv')

    # Compute geometry
    gb.compute_geometry()

    if generate_coarse_grid:
        gb.gen_coarse_grid()

    # Assign parameters
    add_data_to_grid_bucket(gb)

    # Solve system of equations
    problem = pp.problem.Problem(gb)
    solution = problem.solve(solver_name='DualVEMMixDim')

    # Extract and project solution
    primary_var = solution.primary_var
    gb.project_to_grids(primary_var)

    # Export results to VTK file
    pp.exporters.exporter.export_to_vtk(gb, f'results_{description}.vtk')

    # Print information
    print(f'Number of grids: {gb.num_grids}')
    print(f'Number of edges: {gb.num_edges}')
    print(f'Number of faces: {gb.num_faces}')
    print(f'Number of cells: {gb.num_cells}')

def blocking_scenario():
    main(permeability_factor=1e-3, description='blocking', mesh_size=100)

def permeable_scenario():
    main(permeability_factor=1e-1, description='permeable', mesh_size=100, generate_coarse_grid=True)

blocking_scenario()
permeable_scenario()
```