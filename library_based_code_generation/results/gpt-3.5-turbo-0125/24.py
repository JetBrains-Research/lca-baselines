import argparse
import sys
import numpy as np
import matplotlib
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

mesh = fealpy.TriangleMesh(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), np.array([[0, 1, 2], [0, 2, 3]]))
time_sequence = fealpy.UniformTimeLine(0, end_time, time_divisions)

motion_space = fealpy.LagrangeFiniteElementSpace(mesh, motion_degree)
pressure_space = fealpy.LagrangeFiniteElementSpace(mesh, pressure_degree)
N_motion = motion_space.number_of_global_dofs()
N_pressure = pressure_space.number_of_global_dofs()

bilinear_form = PoisuillePDE.BilinearForm(motion_space, pressure_space)
mixed_bilinear_form = PoisuillePDE.MixedBilinearForm(motion_space, pressure_space)

bilinear_form.add_domain_integrator(PoisuillePDE.VelocityStressIntegrator())
mixed_bilinear_form.add_domain_integrator(PoisuillePDE.VelocityStressIntegrator())

A = bilinear_form.assemble()
B = mixed_bilinear_form.assemble()

M = motion_space.mass_matrix()
error_matrix = np.zeros((time_divisions,))

for step in range(steps):
    time_sequence.advance()
    bilinear_form.add_domain_integrator(PoisuillePDE.ScalarConvectionIntegrator(time_sequence))
    A = bilinear_form.assemble()
    
    div = motion_space.divergence_matrix()
    M = motion_space.mass_matrix()
    
    b = np.zeros(N_motion)
    bc = PoisuillePDE.DirichletBC(motion_space, time_sequence.current())
    bc.apply(A, b)
    
    x = mumps.MUMPSContext()
    x.set_operator(A)
    x.solve(b)
    
    motion_space.dof.node[:] = x.get_solution()
    
    error_matrix[step] = np.sqrt(np.sum((motion_space.dof.node - exact_solution)**2))
    
print(np.sum(np.abs(motion_space.dof.node)))