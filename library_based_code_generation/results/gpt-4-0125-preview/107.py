```python
import porepy as pp
import numpy as np
import pandas as pd

def add_data_to_grid_bucket(gb, permeability_factor, aperture, bc_type):
    for g, d in gb:
        kxx = np.ones(g.num_cells) * permeability_factor
        perm = pp.SecondOrderTensor(kxx)
        bc = pp.BoundaryCondition(g, faces=np.array([]), cond=bc_type)
        specified_parameters = {"second_order_tensor": perm, "bc": bc, "aperture": np.ones(g.num_cells) * aperture}
        pp.initialize_data(g, d, "flow", specified_parameters)
    for e, d in gb.edges():
        mg = d["mortar_grid"]
        kxx = np.ones(mg.num_cells) * permeability_factor
        data = {"normal_diffusivity": kxx}
        d[pp.PARAMETERS] = pp.Parameters(mg, ["flow"], {"normal_diffusivity": kxx})
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

def write_network_to_csv(network_string, file_name):
    with open(file_name, 'w') as file:
        file.write(network_string)

def main(permeability_factor, description, mesh_size, generate_coarse_grid=False):
    file_name = "network.csv"
    network_string = "start_x, start_y, end_x, end_y\n0,0,1,1\n1,1,2,2"  # Example network string
    write_network_to_csv(network_string, file_name)
    
    network = pp.fracture_importer.network_2d_from_csv(file_name)
    gb = network.mesh(mesh_args={"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 2})
    gb.compute_geometry()
    
    if generate_coarse_grid:
        coarsener = pp.coarsening.coarsen(gb)
        coarsener.coarsen()
    
    add_data_to_grid_bucket(gb, permeability_factor, 1e-4, "dir")
    
    solver = pp.DualVEMMixDim(gb)
    solver.discretize()
    A, b = solver.assemble_matrix_rhs()
    solution = solver.solve(A, b)
    
    solver.distribute_variable(solution)
    solver.project_flux(solution)
    
    pp.exporter.export_vtk(gb, file_name=description, data_key="flow")
    print(f"Grid bucket with {description}:")
    print(f"Number of grids: {len(gb.grids_of_dimension(2)) + len(gb.grids_of_dimension(1))}")
    print(f"Mesh size: {mesh_size}")

def blocking_scenario():
    main(permeability_factor=1e-12, description="blocking_scenario", mesh_size=0.1)

def permeable_scenario():
    main(permeability_factor=1e-5, description="permeable_scenario", mesh_size=0.2)

# Call the scenarios
blocking_scenario()
permeable_scenario()
```