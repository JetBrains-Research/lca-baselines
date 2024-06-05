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

g = pp.Grid.from_csv("grid.csv")
g.compute_geometry()
if coarsening:
    g.coarsen()
g.assign_node_ordering()

solver = pp.DarcyFlow(gb)
add_data_darcy(gb, tolerance)
solver.solve()
discharge, pressure = solver.extract_solution()

total_flow_rate = pp.fvutils.compute_total_flow_rate(gb)
pp.export_to_vtk(gb, export_folder)

physics = pp.physics.DarcyFlow(gb)
advection_solver = pp.advection.MassMatrixSolver(gb)
add_data_advection(gb, tolerance)
gb.add("time_step", pp.ConstantTimeStepper(time_step_size))

advection_solver.assemble_matrix_rhs()
advection_solver.lu_factorize()

solution = pp.advection.initialize_solution(gb)
times = []
production = []

for _ in range(num_time_steps):
    pp.advection.update_solution(gb)
    production.append(pp.advection.compute_production(gb))
    if _ % export_frequency == 0:
        pp.export_to_vtk(gb, export_folder)

pp.export_times_to_pvd(gb, export_folder)
pp.export_times_and_production_to_txt(times, production)