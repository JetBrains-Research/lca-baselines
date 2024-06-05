  import numpy as np
import scipy as sp
import os
import sys
from porepy import DualVEMMixDim
from porepy import soultz_grid

def add_data_darcy(gb, tol):
    gb.add_parameter('permeability', 1e-12)
    gb.add_parameter('source', 0)
    gb.add_parameter('aperture', 0)
    gb.add_parameter('boundary_conditions', 'Neumann')

def add_data_advection(gb, tol):
    gb.add_parameter('source', 0)
    gb.add_parameter('porosity', 0.5)
    gb.add_parameter('discharge', 0)
    gb.add_parameter('boundary_conditions', 'Neumann')

def create_grid(path, tol):
    sys.path.append(path)
    grid = soultz_grid.Grid(tol)
    grid.create_grid()
    grid.compute_geometry()
    if grid.coarsen_grid():
        grid.assign_node_ordering()
    return grid

def solve_darcy(grid, tol):
    solver = DualVEMMixDim(grid, tol)
    solver.add_parameters()
    solver.compute_matrices()
    solver.compute_rhs()
    solver.solve()
    solver.split_solution()
    solver.extract_discharge()
    solver.project_discharge()
    solver.compute_total_flow_rate()
    return solver

def solve_advection(grid, tol):
    solver = DualVEMMixDim(grid, tol)
    solver.add_parameters()
    solver.compute_matrices()
    solver.compute_rhs()
    solver.solve()
    solver.split_solution()
    solver.extract_discharge()
    solver.project_discharge()
    solver.compute_total_flow_rate()
    return solver

def time_stepping_loop(solver, t_end, dt, export_freq):
    t = 0
    while t < t_end:
        solver.update_solution(dt)
        t += dt
        if t % export_freq == 0:
            solver.export_solution()
    return solver

def main():
    path = 'path/to/grid/file'
    tol = 1e-6
    grid = create_grid(path, tol)
    solver = solve_darcy(grid, tol)
    solver = solve_advection(grid, tol)
    t_end = 100
    dt = 0.01
    export_freq = 10
    solver = time_stepping_loop(solver, t_end, dt, export_freq)
    solver.export_solution()
    solver.export_production_data()

if __name__ == '__main__':
    main()