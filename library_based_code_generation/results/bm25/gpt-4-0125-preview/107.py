```python
import numpy as np
import porepy as pp
import csv

def add_data_to_grid_bucket(gb, permeability_factor):
    for g, d in gb:
        kxx = np.ones(g.num_cells) * permeability_factor
        bc = pp.BoundaryCondition(g)
        bc_val = np.zeros(g.num_faces)
        aperture = np.ones(g.num_cells)
        
        specified_parameters = {"second_order_tensor": pp.SecondOrderTensor(kxx),
                                "bc": bc,
                                "bc_values": bc_val,
                                "aperture": aperture}
        pp.initialize_data(g, d, "flow", specified_parameters)
    
    for e, d in gb.edges():
        mg = d["mortar_grid"]
        kn = np.ones(mg.num_cells) * permeability_factor
        d[pp.PARAMETERS] = pp.Parameters(mg, ["flow"], {"normal_diffusivity": kn})
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

def write_network_to_csv(network_string, file_name):
    network = eval(network_string)
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(network)

def main(permeability_factor, description, mesh_size):
    file_name = "network.csv"
    write_network_to_csv(description, file_name)
    
    network = pp.fracture_importer.network_2d_from_csv(file_name)
    gb = pp.meshing.grid_list_to_grid_bucket(network, mesh_size=mesh_size)
    
    pp.contact_conditions.set_projections(gb)
    pp.contact_conditions.compute_geometry(gb, compute_coarse_grid=True)
    
    add_data_to_grid_bucket(gb, permeability_factor)
    
    solver = pp.DualVEMMixDim(gb)
    solver.discretize()
    solver.solve()
    
    pp.project.project_flux(gb, "flow", "darcy_flux")
    pp.io.to_vtk(gb, "output", "flow")
    
    print(f"GridBucket with {len(gb.grids)} grids and {len(gb.edges())} edges.")

def blocking_scenario():
    permeability_factor = 1e-15
    description = "[((0, 0), (1, 1)), ((1, 0), (0, 1))]"
    mesh_size = 0.1
    main(permeability_factor, description, mesh_size)

def permeable_scenario():
    permeability_factor = 1e-12
    description = "[((0, 0), (1, 1)), ((1, 0), (0, 1))]"
    mesh_size = 0.05
    main(permeability_factor, description, mesh_size)

blocking_scenario()
permeable_scenario()
```