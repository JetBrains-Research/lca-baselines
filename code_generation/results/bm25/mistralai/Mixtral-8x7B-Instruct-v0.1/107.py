 ```python
import porepy as pp
import numpy as np
import csv

def add_data_to_grid_bucket(grid_bucket):
    for grid in grid_bucket:
        permeability = np.random.rand(grid.num_cells)
        apertures = np.random.rand(grid.num_edges)
        bc_type = np.zeros(grid.num_faces, dtype=int)
        bc_value = np.zeros(grid.num_faces)

        grid.props['permeability'] = pp.ScalarField(grid, permeability)
        grid.props['apertures'] = pp.ScalarField(grid, apertures)
        grid.props['boundary_conditions'] = pp.BoundaryCondition(grid, bc_type, bc_value)

        for edge in grid.edges:
            if edge.neighbors[0] is not edge.neighbors[1]:
                coupling_permeability = np.random.rand()
                edge.props['coupling_permeability'] = coupling_permeability

def write_network_to_csv(network, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(network)

def main(permeability_factor, description, mesh_size):
    gb = pp.GridBucket.from_file(f'mesh_{mesh_size}.csv')
    gb.compute_geometry()
    coarse_gb = None if mesh_size == 1 else pp.coarsen_grid_bucket(gb)

    add_data_to_grid_bucket(gb)
    if coarse_gb is not None:
        add_data_to_grid_bucket(coarse_gb)

    problem = pp.DualVEMMixDim(gb, coarse_gb)
    solution = problem.solve()
    extracted_solution = pp.extract_and_project_solution(solution, gb)

    pp.to_vtk(extracted_solution, 'solution', gb)
    print(f'Grid bucket {description} has {gb.num_cells} cells.')

def blocking_scenario():
    main(permeability_factor=0.01, description='blocking', mesh_size=10)

def permeable_scenario():
    main(permeability_factor=1.0, description='permeable', mesh_size=10)

blocking_scenario()
permeable_scenario()
```