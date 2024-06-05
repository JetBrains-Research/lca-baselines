 import fealpy
import numpy as np
import sys
from scipy.sparse.linalg import spsolve
from fealpy.mesh import MeshFactory
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.boundarycondition import DirichletBC
from fealpy.tools import showmultirate

def curl(v):
    return np.array([v[2,:]-v[1,:], v[0,:]-v[2,:], v[1,:]-v[0,:]])

def least_squares_matrix(v, mesh):
    curl_v = curl(v)
    dH = fealpy.functionspace.Hcurl(mesh)
    phi = dH.basis()
    Dphi = dH.grad(phi)
    Psi = dH.grad(phi).T
    return np.einsum('ij,ijk->ik', curl_v, Psi)

def CosSinData(mesh):
    nx, ny = mesh.nnodes
    p = np.zeros((2, nx*ny))
    for i in range(nx):
        for j in range(ny):
            x, y = mesh.node(i, j)
            p[0, i+j*nx] = np.cos(np.pi*x)
            p[1, i+j*nx] = np.sin(np.pi*y)
    return p

def main(degree, h, max_iter, theta):
    # Initialize problem
    pde = fealpy.TimeHarmonicEquation()
    pde.exact_sol = CosSinData

    # Create mesh
    mf = MeshFactory()
    mesh = mf.boxmesh2d(nx=1/h, ny=1/h, layer=0)
    quadrant = int(np.ceil(np.sqrt(nx*ny)/2))
    mesh.remove_element(np.arange(quadrant, nx*ny, step=quadrant))

    # Adaptive iteration
    for i in range(max_iter):
        fes = FirstKindNedelecFiniteElementSpace2d(mesh, degree)
        u = fes.function()

        # Apply boundary conditions
        bc = DirichletBC(fes, pde.exact_sol(mesh.node()))
        bc.apply(u)

        # Solve system of equations
        A = least_squares_matrix(u, mesh)
        b = curl(pde.exact_sol(mesh.node()))
        u.data[:] = spsolve(A, b)

        # Calculate errors
        err_L2 = fealpy.functionspace.L2Error(pde.exact_sol(mesh.node()), u, mesh)
        err_curl = fealpy.functionspace.HcurlError(pde.exact_curl(mesh.node()), curl(u), mesh)
        err_recov = fealpy.functionspace.HcurlError(pde.exact_curl(mesh.node()), least_squares_matrix(u, mesh), mesh)

        print(f'Iteration {i+1}: L2 error = {err_L2:.4e}, curl error = {err_curl:.4e}, recovery error = {err_recov:.4e}')

        # Refine mesh
        if i < max_iter-1:
            mesh.refine_uniform()

    # Plot error rates
    showmultirate(np.array([err_L2, err_curl, err_recov]).T, max_iter, ['L2 error', 'curl error', 'recovery error'])

if __name__ == '__main__':
    degree = int(sys.argv[1])
    h = float(sys.argv[2])
    max_iter = int(sys.argv[3])
    theta = float(sys.argv[4])
    main(degree, h, max_iter, theta)