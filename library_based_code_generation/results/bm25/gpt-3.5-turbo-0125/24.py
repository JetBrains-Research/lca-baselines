import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import mumps
import fealpy

from navier_stokes_mold_2d import PoisuillePDE

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('--motion_degree', type=int, default=1, help='Degree of motion finite element space')
parser.add_argument('--pressure_degree', type=int, default=1, help='Degree of pressure finite element space')
parser.add_argument('--time_divisions', type=int, default=100, help='Number of time divisions')
parser.add_argument('--end_time', type=float, default=1.0, help='Evolution end time')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
parser.add_argument('--steps', type=int, default=10, help='Steps')
parser.add_argument('--nonlinearization_method', type=str, default='Newton', help='Non-linearization method')

args = parser.parse_args()

motion_degree = args.motion_degree
pressure_degree = args.pressure_degree
time_divisions = args.time_divisions
end_time = args.end_time
output_dir = args.output_dir
steps = args.steps
nonlinearization_method = args.nonlinearization_method

mesh = fealpy.unit_square_mesh()
time_sequence = fealpy.UniformTimeLine(0, end_time, time_divisions)

motion_space = fealpy.LagrangeFiniteElementSpace(mesh, motion_degree)
pressure_space = fealpy.LagrangeFiniteElementSpace(mesh, pressure_degree)

Ndof = motion_space.number_of_global_dofs()
Np = pressure_space.number_of_global_dofs()

a = PoisuillePDE()
a.Ndof = Ndof
a.Np = Np

a.init_mesh(mesh)
a.init_time_sequence(time_sequence)
a.init_function_space(motion_space, pressure_space)

a.init_matrix()
a.init_error()

for step in range(steps):
    a.advance()
    a.init_fem_matrix()
    a.init_source_vector()
    a.init_dirichlet()
    a.init_neumann()

    a.solve()
    a.update()
    a.error()
    a.print_error()

print(np.sum(np.abs(a.uh.vector())))