 ```python
from porepy import *
from porepy.mesh import Mesh
from porepy.solvers import DarcySolver, LinearSystemSolver
from porepy.physics import DarcyPhysics, AdvectionPhysics
from porepy.utils import read_mesh_from_csv

def add_data_darcy(gb, tol, permeability, source, aperture, bc):
    physics = DarcyPhysics(tol)
    physics.set_permeability(permeability)
    physics.set_source(source)
    physics.set_aperture(aperture)
    physics.set_boundary_conditions(bc)
    gb.add_physics(physics)

def add_data_advection(gb, tol, porosity, discharge, bc):
    physics = AdvectionPhysics(tol)
    physics.set_porosity(porosity)
    physics.set_discharge(discharge)
    physics.set_boundary_conditions(bc)
    gb.add_physics(physics)

tol = 1e-8
export_folder = "results"
time = 10.0
num_time_steps = 100
time_step_size = time / num_time_steps
export_frequency = 10
coarsening = True

mesh_size = {1: 10, 2: 5, 3: 2}
domain_boundaries = {
    (1, 2): Line,
    (2, 3): Line,
}

mesh = read_mesh_from_csv("mesh.csv")
mesh.compute_geometry()
if coarsening:
    mesh.coarsen()
mesh.assign_node_ordering()

gb = mesh.get_grid_bucket()
darcy_solver = DarcySolver()
darcy_physics = DarcyPhysics(tol)
darcy_physics.set_permeability(1.0)
darcy_physics.set_source(0.0)
darcy_physics.set_aperture(1.0)
darcy_physics.set_boundary_conditions({
    (1, 2): Dirichlet((0.0, 0.0)),
    (2, 3): Dirichlet((0.0, 0.0)),
})
gb.add_physics(darcy_physics)
gb.add_solver(darcy_solver)

darcy_solver.solve()
discharge, pressure = darcy_solver.get_discharge_and_pressure()
total_flow_rate = discharge.integrate()

write_vtk(export_folder, "darcy", mesh, discharge, pressure)

advection_physics = AdvectionPhysics(tol)
advection_physics.set_porosity(1.0)
advection_physics.set_discharge(discharge)
advection_physics.set_boundary_conditions({
    (1, 2): Dirichlet(0.0),
    (2, 3): Dirichlet(0.0),
})
gb.add_physics(advection_physics)

mass_matrix_solver = LinearSystemSolver()
advection_matrix, _ = advection_physics.get_mass_matrix_and_rhs()
mass_matrix_solver.solve(advection_matrix, advection_physics.get_rhs())
advection_matrix.lu_factorize()

solution = zeros(len(mesh.nodes))
time_steps = zeros(num_time_steps)
production = zeros(num_time_steps)

for i in range(num_time_steps):
    advection_rhs = advection_physics.get_rhs()
    advection_physics.set_source(discharge)
    advection_physics.set_boundary_conditions({
        (1, 2): Dirichlet(discharge[1]),
        (2, 3): Dirichlet(discharge[-1]),
    })
    advection_physics.set_discharge(None)
    advection_physics.get_rhs(solution)
    solution += time_step_size * advection_rhs

    if i % export_frequency == 0:
        write_vtk(export_folder, f"advection_{i}", mesh, solution, None)

    time_steps[i] = time * i / num_time_steps
    production[i] = discharge.integrate()

write_pvd(export_folder, "advection", time_steps, solution)
with open(f"{export_folder}/production.txt", "w") as f:
    f.write(f"Time\tProduction\n")
    for t, p in zip(time_steps, production):
        f.write(f"{t}\t{p}\n")
```