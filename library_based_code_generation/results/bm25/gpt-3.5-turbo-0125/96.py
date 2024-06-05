```python
import porepy as pp

def add_data_darcy(gb, tol):
    pass

def add_data_advection(gb, tol):
    pass

tolerance = 1e-6
export_folder = "results/"
time = 10
num_time_steps = 100
time_step_size = time / num_time_steps
export_frequency = 10
coarsening = False

mesh_size = {"domain": 1.0}
domain_boundaries = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

grid = pp.Grid.from_csv("grid.csv")
grid.compute_geometry()
if coarsening:
    grid.coarsen()
grid.assign_node_ordering()

darcy_solver = pp.DarcySolver()
add_data_darcy(grid, tolerance)
darcy_solver.solve(grid)
discharge, pressure = darcy_solver.extract_solution()
total_flow_rate = darcy_solver.compute_total_flow_rate()
pp.export_to_vtk(grid, export_folder)

physics = pp.Physics()
advection_solver = pp.AdvectionSolver()
mass_matrix_solver = pp.MassMatrixSolver()
add_data_advection(grid, tolerance)
grid.add_time_step_property(time_step_size)

advection_matrices, advection_rhs = advection_solver.create_matrices_and_rhs(grid)
mass_matrix = mass_matrix_solver.create_mass_matrix(grid)
lu_factorization = pp.LU_factorization()

solution = pp.initialize_solution_vector(grid)
time_array = []
production_array = []

for i in range(num_time_steps):
    solution = advection_solver.update_solution(grid, solution)
    production = advection_solver.compute_production(grid, solution)
    if i % export_frequency == 0:
        pp.export_to_vtk(grid, export_folder)
    time_array.append(i * time_step_size)
    production_array.append(production)

pp.export_time_steps_to_pvd(time_array, export_folder)
pp.save_to_txt(time_array, production_array, export_folder)
```