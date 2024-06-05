 ```python
import numpy as np
from scipy.sparse import linalg
from porepy import TestGettersAndSetters, DarcyAndTransport, GridBucket, GmshGridBucketWriter

def add_data(grid_bucket, domain, permeability_factor):
    permeability = permeability_ * np.ones(domain.num_cells)
    source_term = np.zeros(domain.num_cells)
    apertures = np.ones(domain.num_edges)
    boundary_conditions = {'left': 1.0, 'right': 0.0}

    grid_bucket.set_param('permeability', permeability)
    grid_bucket.set_param('source_term', source_term)
    grid_bucket.set_param('apertures', apertures)
    grid_bucket.set_param('boundary_conditions', boundary_conditions)

def write_network(file_name):
    with open(file_name, 'w') as f:
        f.write('''
L 0 0 10 0 10 10 0 10 0 0
C 0 0 1 0 1 1 0 1 0 0
C 0 0 0 1 1 1 0 1 0 0
C 0 0 0 1 0 0 0 1 0 0
C 10 10 10 0 10 0 10 0 10 10
C 10 10 10 1 10 1 10 1 10 10
C 10 10 9 1 9 0 9 0 9 10
C 10 10 9 0 9 0 9 1 9 0
        ''')

def main(permeability_factor, description, coarsen, export):
    dim = 2
    length = 10
    num_cells = 10
    num_fractures = 4

    domain = GridBucket(dim, length, num_cells, num_fractures)

    file_name = _make_file_name(description)
    GmshGridBucketWriter(domain).write_to_file(file_name)

    snapped_grid = snap_to_grid(file_name, domain)

    permeability = permeability_ * np.ones(domain.num_cells)
    snapped_grid.set_param('permeability', permeability)

    if coarsen:
        snapped_grid.coarsen()

    ordering = np.arange(snapped_grid.num_nodes)
    snapped_grid.set_param('ordering', ordering)

    add_data(snapped_grid, domain, permeability_factor)

    problem = DarcyAndTransport(domain)
    problem.compute_geometry()

    solver_flow = linalg.SuperLU()
    solver_source = linalg.SuperLU()

    for sb in snapped_grid.subbuckets:
        A, b = assemble_in_bucket(sb, problem, solver_flow)
        x = solver_flow.solve(A, b)
        sb.params['solution_flow'] = x

        A, b = assemble_in_bucket(sb, problem, solver_source)
        x = solver_source.solve(A, b)
        sb.params['solution_source'] = x

    for sb in snapped_grid.subbuckets:
        solution = sb.params['solution_flow']
        sb.project_to_subbucket(solution)

    discharge = snapped_grid.get_param('discharge')
    pressure = snapped_grid.get_param('pressure')

    if export:
        TestGettersAndSetters.export_to_vtk(file_name, pressure, discharge)

def test_vem_blocking():
    main(0.001, 'blocking', False, True)

def test_vem_permeable():
    main(10, 'permeable', False, True)
```