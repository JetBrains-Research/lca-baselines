 ```python
import numpy as np
from scipy.sparse import csr_matrix
from porepy import Bucket, Properties, Field, Solver, ElemProperties
from porepy.models.flow import Flow
from porepy.utils import generate_network_string

def add_data(grid_bucket, domain, permeability_factor):
    # Permeability
    permeability = permeability_factor * np.ones(domain.num_cells)
    grid_bucket.add_param('permeability', permeability)

    # Source term
    source_term = np.zeros(domain.num_cells)
    grid_bucket.add_param('source_term', source_term)

    # Apertures
    apertures = np.ones(domain.num_faces)
    grid_bucket.add_param('apertures', apertures)

    # Boundary conditions
    bc_type = Properties(name='bc_type')
    bc_type.add_prop('left', 'dirichlet', 0)
    bc_type.add_prop('right', 'neumann', 0)
    grid_bucket.add_param('bc_type', bc_type)

def write_network(file_name):
    network_string = '''
    1 2 3 4
    2 5 6
    3 7 8
    4 9 10
    5 11
    6 12
    7 13
    8 14
    9 15
    10 16
    '''
    with open(file_name, 'w') as file:
        file.write(network_string)

def main(permeability_factor, description, coarsen, export):
    # Mesh parameters
    dim = 2
    length = 1
    num_cells = (10, 10)
    num_faces = None
    num_edges = None
    num_vertices = None
    bc_vertices = {'left': [0], 'right': [num_cells[0]-1]}

    # Domain
    domain = Bucket(dim=dim, num_cells=num_cells, num_faces=num_faces,
                    num_edges=num_edges, num_vertices=num_vertices,
                    bc_vertices=bc_vertices)

    # File name
    file_name = f'fracture_network_{description}.txt'

    # Write network to file
    write_network(file_name)

    # Import 2D fracture network from file
    Flow.import_2d_fracture_network(domain, file_name)

    # Compute geometry of grid bucket
    domain.compute_geometry()

    # Optionally coarsen grid bucket
    if coarsen:
        domain.coarsen()

    # Assign ordering to nodes of grid bucket
    domain.assign_ordering()

    # Add data to grid bucket
    add_data(domain.grid_buckets[0], domain, permeability_factor)

    # Define solvers for flow and source
    flow_solver = Solver(Flow(domain=domain))
    source_solver = Solver(Flow(domain=domain))

    # Compute right-hand side and matrix of linear systems
    rhs, mat = flow_solver.compute_rhs_and_mat()
    source_rhs, source_mat = source_solver.compute_rhs_and_mat()

    # Solve linear systems
    flow_solution = flow_solver.solve(rhs, mat)
    source_solution = source_solver.solve(source_rhs, source_mat)

    # Split solution
    pressure = Field(name='pressure', values=flow_solution)
    discharge = Field(name='discharge', values=source_solution)

    # Extract discharge and pressure from solution
    domain.extract_fields([discharge, pressure])

    # Project discharge
    discharge_proj = ElemProperties(name='discharge_proj')
    for elem in domain.elem_iter(scaling=1):
        discharge_proj.add_prop(elem.index, elem.centroid * discharge[elem])
    domain.add_param('discharge_proj', discharge_proj)

    # Optionally export results to vtk file
    if export:
        domain.write_vtk(f'results_{description}.vtk')

def test_vem_blocking():
    main(permeability_factor=0.001, description='blocking', coarsen=False, export=True)

def test_vem_permeable():
    main(permeability_factor=100, description='permeable', coarsen=False, export=True)
```