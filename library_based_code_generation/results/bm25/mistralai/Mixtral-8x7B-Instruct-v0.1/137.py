 ```python
import os
import sys
import numpy as np
from scipy.sparse import linalg
from porepy import (
    DarcyAndTransport,
    test_ismember_rows_double_occurence_a_and_b,
    _a_in_b,
    test_round_up_and_down,
    _point_and_edge_lists,
    TestGettersAndSetters,
    _uniquify_points_and_edges,
    test_many_points_and_segments,
    test_single_point_and_segment,
    test_one_cell_a_time_node_keyword,
    Parameters,
    snap_to_grid,
    check_parameters,
    map_subgrid_to_grid,
    SnapToGridTest,
    test_ismember_rows_double_occurence_a_no_b,
    apply_function_to_edges,
    apply_function_to_nodes,
    to_vtk,
    to_gmsh,
)

sys.path.append("path/to/soultz_grid")
import soultz_grid


def add_data_darcy(gb, tol):
    gb.add_param("perm_x", np.ones(gb.num_cells), "cell", "darcy", "permeability_x")
    gb.add_param("perm_y", np.ones(gb.num_cells), "cell", "darcy", "permeability_y")
    gb.add_param("perm_z", np.ones(gb.num_cells), "cell", "darcy", "permeability_z")
    gb.add_param("source", np.zeros(gb.num_cells), "cell", "darcy", "source_term")
    gb.add_param("aperture", np.ones(gb.num_cells), "cell", "darcy", "aperture")

    gb.add_bc("dirichlet_left", "cell", "darcy", "strong", "pressure", 1)
    gb.add_bc("dirichlet_right", "cell", "darcy", "strong", "pressure", 0)
    gb.add_bc("neumann_top", "cell", "darcy", "weak", "discharge", 0)
    gb.add_bc("neumann_bottom", "cell", "darcy", "weak", "discharge", 0)

    gb.add_bc("dirichlet_left", "face", "darcy", "strong", "pressure", 1)
    gb.add_bc("dirichlet_right", "face", "darcy", "strong", "pressure", 0)
    gb.add_bc("neumann_top", "face", "darcy", "weak", "discharge", 0)
    gb.add_bc("neumann_bottom", "face", "darcy", "weak", "discharge", 0)


def add_data_advection(gb, tol):
    gb.add_param("source", np.zeros(gb.num_cells), "cell", "transport", "source_term")
    gb.add_param("porosity", np.ones(gb.num_cells), "cell", "transport", "porosity")
    gb.add_param("discharge", np.zeros(gb.num_cells), "cell", "transport", "discharge")

    gb.add_bc("dirichlet_left", "cell", "transport", "strong", "concentration", 1)
    gb.add_bc("dirichlet_right", "cell", "transport", "strong", "concentration", 0)

    gb.add_bc("dirichlet_left", "face", "transport", "strong", "concentration", 1)
    gb.add_bc("dirichlet_right", "face", "transport", "strong", "concentration", 0)


gb = DarcyAndTransport(dim=3)

add_data_darcy(gb, 1e-12)
add_data_advection(gb, 1e-12)

params = Parameters()
params.set("tolerance", 1e-12)
params.set("snap_to_grid", True)

grid = soultz_grid.create_grid(params)

geometry = grid.compute_geometry()

if geometry.max_cell_diameter > 10 * geometry.min_cell_diameter:
    grid = grid.coarsen()

grid.set_node_ordering()

darcy_solver = grid.solve(
    "darcy",
    solver="DualVEMMixDim",
    parameters={"use_cell_centers": True, "use_mixed_formulation": True},
)

pressure, discharge = darcy_solver.split()

discharge = snap_to_grid(discharge, grid, "face", "darcy")

total_flow_rate = discharge.integrate(
    dim=geometry.dim,
    boundary=True,
    boundary_marker="neumann_top",
)

transport_params = Parameters()
transport_params.set("transport_solver", "explicit_euler")

advection_solver = grid.solve(
    "transport",
    solver="Advection",
    parameters=transport_params,
)

mass_matrix_solver = grid.solve(
    "transport",
    solver="MassMatrix",
    parameters=transport_params,
)

transport_params.set("mass_matrix_solver", mass_matrix_solver)
transport_params.set("advection_solver", advection_solver)

gb.add_subgrid_data(grid, "darcy")
gb.add_subgrid_data(grid, "transport")

matrices, rhs = gb.get_matrices_and_vectors(
    "transport",
    subgrid_marker="darcy",
    solver_name="mass_matrix",
)

LU = linalg.splu(matrices[0])

sol = np.zeros(gb.num_cells)

dt = 0.1
time = 0

while time < 10:
    LU.solve(rhs, sol)
    time += dt
    gb.set_param("concentration", sol, "cell", "transport")

    matrices, rhs = gb.get_matrices_and_vectors(
        "transport",
        subgrid_marker="darcy",
        solver_name="advection",
    )

    LU.solve(rhs, sol)
    time += dt
    gb.set_param("concentration", sol, "cell", "transport")

gb.set_param("concentration", sol, "cell", "transport")

to_vtk(gb, "output.vtk", point_data=["pressure", "concentration"])

production_data = {"total_flow_rate": total_flow_rate, "time": time}
np.savetxt("production_data.txt", list(production_data.items()))
```