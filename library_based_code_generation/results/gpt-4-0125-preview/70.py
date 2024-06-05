```python
import numpy as np
from scipy.sparse.linalg import spsolve
import porepy as pp

def add_data(grid_bucket, domain, permeability_factor):
    for g, d in grid_bucket:
        # Assign permeability
        perm = pp.SecondOrderTensor(np.ones(g.num_cells) * permeability_factor)
        d[pp.PARAMETERS] = pp.Parameters(g)
        d[pp.PARAMETERS]["second_order_tensor"] = perm
        
        # Assign source term
        source_term = np.zeros(g.num_cells)
        d[pp.PARAMETERS]["source"] = source_term
        
        # Assign apertures
        if g.dim < grid_bucket.dim_max():
            aperture = np.power(0.1, g.dim)
            d[pp.PARAMETERS]["aperture"] = np.ones(g.num_cells) * aperture
        
        # Assign boundary conditions
        bc_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_type = np.array(["dir"] * bc_faces.size)
        bc_values = np.zeros(g.num_faces)
        bc_values[bc_faces] = 1.0
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        d[pp.PARAMETERS]["bc"] = bc
        d[pp.PARAMETERS]["bc_values"] = bc_values

def write_network(file_name):
    network_string = "1 0 0 1 0\n0 1 0 0 1\n"
    with open(file_name, "w") as file:
        file.write(network_string)

def main(permeability_factor, description, coarsen, export_results):
    mesh_kwargs = {"mesh_size_frac": 0.2, "mesh_size_min": 0.1}
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    file_name = "network.txt"
    
    write_network(file_name)
    
    network = pp.fracture_importer.network_2d_from_csv(file_name)
    gb = pp.meshing.cart_grid([network], nx=[4, 4], physdims=[1, 1])
    gb.compute_geometry()
    
    if coarsen:
        coarsener = pp.coarsening.coarsen(gb)
        coarsener.coarsen()
    
    pp.contact_conditions.set_projections(gb)
    add_data(gb, domain, permeability_factor)
    
    solver_flow = pp.Tpfa("flow")
    solver_source = pp.ScalarSource("flow")
    
    A, b = solver_flow.assemble_matrix_rhs(gb)
    x = spsolve(A, b)
    
    solver_flow.distribute_variable(gb, solution_vector=x)
    pressure = solver_flow.extract_pressure(gb, x)
    discharge = solver_flow.extract_flux(gb)
    
    if export_results:
        pp.exporter.export_vtk(gb, file_name=description, data={"pressure": pressure, "discharge": discharge})

def test_vem_blocking():
    main(0.001, "blocking", False, True)

def test_vem_permeable():
    main(1000, "permeable", False, True)
```