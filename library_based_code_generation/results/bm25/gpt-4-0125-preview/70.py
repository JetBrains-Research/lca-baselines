```python
import numpy as np
from scipy.sparse.linalg import spsolve
import porepy as pp

def add_data(grid_bucket, domain, permeability_factor):
    for g, d in grid_bucket:
        # Assign permeability
        kxx = np.ones(g.num_cells) * permeability_factor
        perm = pp.SecondOrderTensor(kxx)
        d[pp.PARAMETERS] = {"second_order_tensor": perm}

        # Assign source term
        f = np.ones(g.num_cells)
        d[pp.PARAMETERS]["source"] = f

        # Assign apertures for fractures
        if g.dim < grid_bucket.dim_max():
            aperture = np.power(0.1, g.dim)
            d[pp.PARAMETERS]["aperture"] = aperture

        # Assign boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(["dir"] * bc_faces.size)
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_values = np.zeros(g.num_faces)
        bc_values[bc_faces] = 1
        d[pp.PARAMETERS]["bc"] = bc
        d[pp.PARAMETERS]["bc_values"] = bc_values

def write_network(file_name):
    network_string = "1 0 0 0\n0 1 0 0\n2\n0 1\n"
    with open(file_name, "w") as file:
        file.write(network_string)

def main(permeability_factor, description, coarsen, export_results):
    mesh_kwargs = {"mesh_size_frac": 0.1, "mesh_size_min": 0.02}
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
    pp.contact_conditions.set_permeability_from_aperture(gb)
    
    add_data(gb, domain, permeability_factor)
    
    flow_discretization = pp.Tpfa("flow")
    source_discretization = pp.ScalarSource("flow")
    
    assembler = pp.Assembler(gb)
    assembler.discretize()
    
    A, b = assembler.assemble_matrix_rhs()
    solution = spsolve(A, b)
    
    assembler.distribute_variable(solution)
    
    if export_results:
        pp.Exporter(gb, file_name=description, folder_name="results").write_vtk(["pressure", "flux"])

def test_vem_blocking():
    main(0.001, "blocking", False, True)

def test_vem_permeable():
    main(1000, "permeable", False, True)
```