 ```python
import porepy as pp
import numpy as np
import time
import os
import csv

def add_data_darcy(gb, tol):
    for dim in [2, 3]:
        for sb in gb.subgrids(dim):
            if not sb.is_root:
                continue

            # Permeability
            K = sb.create_data("darcy_K", () => 1e-12)

            # Source term
            sb.create_data("darcy_source", () => 0.0)

            # Aperture
            sb.create_data("darcy_aperture", () => 1.0)

            # Boundary conditions
            sb.create_data("darcy_bc", () => 0)

def add_data_advection(gb, tol):
    for dim in [2, 3]:
        for sb in gb.subgrids(dim):
            if not sb.is_root:
                continue

            # Source term
            sb.create_data("adv_source", () => 0.0)

            # Porosity
            sb.create_data("adv_porosity", () => 1.0)

            # Discharge
            sb.create_data("adv_discharge", () => 0.0)

            # Boundary conditions
            sb.create_data("adv_bc", () => 0)

tol = 1e-14
exp_folder = "outputs"
time_max = 10
n_time_steps = 100
time_step_size = 0.1
exp_frequency = 10
coarsening = False

mesh_sizes = {2: 50, 3: 20}
domain_boundaries = {2: [1, 2, 3, 4], 3: [1, 2, 3, 4, 5, 6]}

gb = pp.GridBuilder(2).build_grid(os.path.join(exp_folder, "grid.csv"))
gb.compute_geometry()

if coarsening:
    gb.coarsen_max(tol)
gb.assign_node_ordering()

darcy_solver = pp.DarcySolver(gb)

for sb in gb.subgrids():
    add_data_darcy(sb.grid_bucket, tol)

darcy_prob = pp.DarcyProblem(gb, darcy_solver)
darcy_prob.solve()

discharge = darcy_prob.get_discharge()
pressure = darcy_prob.get_pressure()

total_flow_rate = discharge.integrate(gb.total_volume())

pp.export_to_vtk(os.path.join(exp_folder, "darcy_solution.vtk"), gb, {"pressure": pressure, "discharge": discharge})

physics = pp.List(["darcy", "advection"])
adv_solver = pp.AdvectionSolver(gb)
mass_solver = pp.MassMatrixSolver(gb)

for sb in gb.subgrids():
    add_data_advection(sb.grid_bucket, tol)

gb.time_step_property.set_value(0)

for t in range(n_time_steps):
    time_step = time_step_size * t

    if time_step > time_max:
        break

    for sb in gb.subgrids():
        sb.time_step_property.set_value(time_step)

    for p in physics:
        if p == "darcy":
            darcy_prob.solve()
            discharge = darcy_prob.get_discharge()
            pressure = darcy_prob.get_pressure()

        if p == "advection":
            advection_prob = pp.AdvectionProblem(gb, adv_solver)
            advection_prob.solve()

            mass_prob = pp.MassMatrixProblem(gb, mass_solver)
            mass_prob.solve()

    solution = {"pressure": pressure, "discharge": discharge}

    if (t + 1) % exp_frequency == 0:
        pp.export_to_vtk(os.path.join(exp_folder, f"time_{t}.vtk"), gb, solution)

pp.export_to_pvd(os.path.join(exp_folder, "time_series.pvd"), gb)

with open(os.path.join(exp_folder, "production.txt"), "w") as f:
    f.write("Time\tProduction\n")
    for t in range(n_time_steps):
        time_step = time_step_size * t
        f.write(f"{time_step}\t{total_flow_rate}\n")
```