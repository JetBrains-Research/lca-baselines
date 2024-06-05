 ```python
import porepy as pp
import numpy as np
import os

def add_data_darcy(gb, tol):
    gb.add_property('darcy_permeability', np.random.rand(gb.dim, gb.dim))
    gb.add_property('darcy_source', np.zeros(gb.num_cells))
    gb.add_property('darcy_aperture', np.random.rand(gb.num_faces))
    gb.add_property('darcy_boundary_conditions', np.zeros(gb.num_faces, dtype=int))

def add_data_advection(gb, tol):
    gb.add_property('advection_source', np.zeros(gb.num_cells))
    gb.add_property('advection_porosity', np.random.rand(gb.num_cells))
    gb.add_property('advection_discharge', np.zeros(gb.num_faces))
    gb.add_property('advection_boundary_conditions', np.zeros(gb.num_faces, dtype=int))

tol = 1e-12
exp_folder = 'outputs'
time_max = 10
n_time_steps = 100
time_step_size = 0.1
exp_frequency = 10
coarsening = False

mesh_sizes = {2: 0.1, 3: 0.01}
domain_boundaries = {2: [1, 2], 3: [1, 2, 3, 4]}

gb = pp.GridBucket(2).create_grid_2d_from_csv('mesh.csv')
gb.compute_geometry()
if coarsening:
    gb = pp.coarsen(gb, tol)
gb.assign_node_ordering()

darcy_solver = pp.DarcyAndTransport(gb)
add_data_darcy(gb, tol)
darcy_prob = pp.DarcyProblem(gb, darcy_solver)
darcy_prob.solve()
discharge = darcy_prob.get_property('discharge')
pressure = darcy_prob.get_property('pressure')
total_flow_rate = np.sum(discharge)
pp.to_vtk(gb, os.path.join(exp_folder, 'darcy_discharge.vtk'), 'discharge', discharge)
pp.to_vtk(gb, os.path.join(exp_folder, 'darcy_pressure.vtk'), 'pressure', pressure)

physics = pp.List(pp.DarcyProblem(gb, darcy_solver))
advection_solver = pp.AdvectionSolver(gb)
mass_matrix_solver = pp.MassMatrixSolver(gb)
add_data_advection(gb, tol)
gb.add_property('time_step_size', time_step_size)

for time_step in range(n_time_steps):
    time = time_step * time_step_size
    if time > time_max:
        break

    matrices, rhs = advection_solver.create_matrices_and_rhs(gb, physics)
    advection_solver.lu_factorize(matrices)
    solution = advection_solver.solve(matrices, rhs)

    mass_matrix_solver.create_mass_matrix(gb)
    mass_matrix_solver.compute_diagonal(gb)
    mass_matrix_solver.apply_mass_matrix(gb, solution)

    production = np.sum(solution * discharge)

    if (time_step + 1) % exp_frequency == 0:
        pp.to_vtk(gb, os.path.join(exp_folder, f'advection_solution_{time_step}.vtk'), 'solution', solution)

pp.to_pvd(gb, os.path.join(exp_folder, 'advection_solution.pvd'), exp_frequency)
with open(os.path.join(exp_folder, 'production.txt'), 'w') as f:
    f.write('Time Production\n')
    for time_step in range(n_time_steps):
        time = time_step * time_step_size
        if time > time_max:
            break
        f.write(f'{time} {production}\n')
```