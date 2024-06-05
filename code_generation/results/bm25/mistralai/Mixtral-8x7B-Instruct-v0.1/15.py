 import fealpy
import numpy as np
import sys
from scipy.sparse.linalg import spsolve
from fealpy.mesh import MeshFactory
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.boundarycondition import DirichletBC
from fealpy.tools import curl_solution, curl_curl_solution
from fealpy.plot import showmultirate

def least_squares_matrix(Vh):
    darts = Vh.number_of_darts()
    L = np.zeros((darts, darts))
    for i in range(darts):
        for j in range(darts):
            L[i, j] = np.dot(Vh.grad(i), Vh.grad(j))
    return L

def recover_curl(Vh, uh):
    curl_uh = np.zeros(Vh.number_of_nodes_of_all_cells())
    for i in range(Vh.number_of_cells()):
        curl_uh[Vh.number_of_nodes_of_cells(i):Vh.number_of_nodes_of_cells(i+1)] = np.cross(Vh.edge(i, 0), Vh.edge(i, 1)) \
                                                                                * uh[Vh.number_of_basis(i):Vh.number_of_basis(i+1)].sum()
    return curl_uh

def time_harmonic_equation(degree, h, max_iter, theta):
    data = fealpy.data.CosSinData()
    mesh = MeshFactory().box(xmin=0, xmax=1, ymin=0, ymax=1)
    mesh.remove_quadrant(3)
    fes = FirstKindNedelecFiniteElementSpace2d(mesh, degree)

    uh = np.zeros(fes.number_of_dofs())
    bc = DirichletBC(fes, lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]), boundary_marker=1)

    errors = []
    for iter in range(max_iter):
        L = least_squares_matrix(fes)
        A = fes.assemble_matrix(L)
        b = fes.assemble_vector(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        bc.apply(A, b)
        uh[:] = spsolve(A, b)

        err_L2 = np.linalg.norm(data.solution - uh, ord=2) / np.linalg.norm(data.solution, ord=2)
        err_curl = np.linalg.norm(data.curl - curl_solution(fes, uh), ord=2) / np.linalg.norm(data.curl, ord=2)
        err_curl_recover = np.linalg.norm(data.curl - recover_curl(fes, uh), ord=2) / np.linalg.norm(data.curl, ord=2)

        print(f"Iter {iter}: L2 error = {err_L2:.4e}, curl error = {err_curl:.4e}, curl recover error = {err_curl_recover:.4e}")

        if iter < max_iter - 1:
            errors.append([err_L2, err_curl, err_curl_recover])
            fes.refine(theta)

    showmultirate(np.array(errors), ['L2', 'curl', 'curl_recover'])

if __name__ == '__main__':
    degree = int(sys.argv[1])
    h = float(sys.argv[2])
    max_iter = int(sys.argv[3])
    theta = float(sys.argv[4])

    time_harmonic_equation(degree, h, max_iter, theta)