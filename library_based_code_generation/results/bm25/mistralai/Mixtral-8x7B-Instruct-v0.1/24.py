 import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from mumps import *
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.timeintegrator import UniformTimeLine
from fealpy.pde.navier_stokes_mold_2d import PoisuillePDE

def create_mesh():
    return ATriMesher.unit_square_mesh(10, 10)

def create_timeline(nt, dt):
    return UniformTimeLine(nt, dt)

def create_spaces(mesh, p, q):
    return [LagrangeFiniteElementSpace(mesh, p), LagrangeFiniteElementSpace(mesh, q)]

def get_number_of_dofs(spaces):
    return [space.number_of_global_dofs for space in spaces]

def create_bilinear_form(spaces):
    return [BilinearForm(space) for space in spaces]

def create_mixed_bilinear_form(spaces):
    return MixedBilinearForm(spaces)

def add_domain_integrators(biforms, spaces, pde):
    for biform, space, p in zip(biforms, spaces, pde.motion_domains):
        p.add_domain_integrator(biform, space)

def add_mixed_domain_integrators(mbiform, spaces, pde):
    pde.add_mixed_domain_integrator(mbiform, spaces)

def assemble_matrices(biforms, mbiform, spaces):
    A = [biform.assemble_matrix() for biform in biforms]
    M = mbiform.assemble_matrix()
    return A, M

def calculate_mass_matrix(biform, space):
    return biform.assemble_matrix(is_assembled=False, is_need_dof_coordinate=True)

def initialize_error_matrix(space):
    return np.zeros(space.number_of_global_dofs, dtype=np.float64)

def advance_time_level(timeline, biforms, mbiform, spaces, A, M, mass_matrix, error_matrix, pde, u, v, dt):
    for t in timeline.next_time_level():
        biform = BilinearForm(spaces[0])
        pde.add_convection_integrator(biform, spaces, u, dt)
        A_new = biform.assemble_matrix()
        M_new = calculate_mass_matrix(biform, spaces[0])
        divergence_matrix = pde.divergence_matrix(spaces)
        source_vector = pde.source_vector(spaces, t)
        boundary_conditions = pde.boundary_conditions(spaces, t)

        M_new = M_new + dt * divergence_matrix.T @ M_new
        M_new = M_new.tocsr()
        M_new_inv = spsolve(M_new, np.ones(M_new.shape[0]))
        M_new_inv = M_new_inv.reshape(-1, 1)
        A_new = A_new @ M_new_inv
        source_vector = source_vector @ M_new_inv

        u_new = spsolve(A_new, source_vector)
        v_new = spsolve(M, u_new)

        error_matrix[:] = 0
        error_matrix[:spaces[0].number_of_global_dofs] = u_new - u
        error_matrix[spaces[0].number_of_global_dofs:] = v_new - v

        u = u_new
        v = v_new

def solve_system(A, b):
    return spsolve(A, b)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=1, help='degree of motion finite element space')
    parser.add_argument('--q', type=int, default=1, help='degree of pressure finite element space')
    parser.add_argument('--nt', type=int, default=10, help='number of time divisions')
    parser.add_argument('--end_time', type=float, default=1.0, help='evolution end time')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--steps', type=int, default=10, help='number of steps')
    parser.add_argument('--nonlinearization', type=str, default='explicit', help='non-linearization method')
    args = parser.parse_args()

    mesh = create_mesh()
    timeline = create_timeline(args.nt, args.end_time / args.nt)
    spaces = create_spaces(mesh, args.p, args.q)
    biforms = create_bilinear_form(spaces)
    mbiform = create_mixed_bilinear_form(spaces)

    pde = PoisuillePDE(mesh, args.p, args.q, args.end_time, args.nonlinearization)
    add_domain_integrators(biforms, spaces, pde)
    add_mixed_domain_integrators(mbiform, spaces, pde)

    A, M = assemble_matrices(biforms, mbiform, spaces)
    mass_matrix = [calculate_mass_matrix(biform, space) for biform, space in zip(biforms, spaces)]
    error_matrix = [initialize_error_matrix(space) for space in spaces]

    u = np.zeros(spaces[0].number_of_global_dofs)
    v = np.zeros(spaces[1].number_of_global_dofs)

    for i in range(args.steps):
        advance_time_level(timeline, biforms, mbiform, spaces, A, M, mass_matrix, error_matrix, pde, u, v, args.end_time / args.nt)

    print(np.sum(np.abs(u)))