import numpy as np
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.mesh import MeshFactory
from fealpy.boundarycondition import DirichletBC
from fealpy.errornorm import L2_error
from fealpy.show import showmultirate
from fealpy.functionspace.errornorm import L2_error
from scipy.sparse.linalg import spsolve

def recover_curl(uh):
    pass

def least_squares_matrix(mesh, uh):
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Solve time-harmonic equation using adaptive methods')
    parser.add_argument('--degree', type=int, default=1, help='Degree of the first kind Nedelec element')
    parser.add_argument('--meshsize', type=float, default=0.1, help='Initial mesh size')
    parser.add_argument('--maxiter', type=int, default=5, help='Maximum number of adaptive iterations')
    parser.add_argument('--theta', type=float, default=0.5, help='Theta parameter for adaptive iteration')
    args = parser.parse_args()

    data = CosSinData()
    mesh = MeshFactory.boxmesh2d([0, 1, 0, 1], nx=4, ny=4, meshtype='tri')
    mesh.delete(np.arange(4))

    for i in range(args.maxiter):
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=args.degree)
        bc = DirichletBC(space, f=data.f, g=data.g, is_dirichlet_boundary=mesh.ds.boundary('all'))
        A, b = space.assemble_system(data.pde, dirichlet=bc, return_matrix=True)
        uh = spsolve(A, b)
        
        error = L2_error(space, data.u, uh)
        curl_error = L2_error(space, data.curl, recover_curl(uh))
        recovery_error = L2_error(space, recover_curl(uh), least_squares_matrix(mesh, uh) @ uh)
        
        if i < args.maxiter - 1:
            eta = recovery_error
            markedCell = mesh.adaptive_mark(eta, theta=args.theta)
            mesh.refine(markedCell)

    showmultirate(plt, 0, error, errorType='L2')
