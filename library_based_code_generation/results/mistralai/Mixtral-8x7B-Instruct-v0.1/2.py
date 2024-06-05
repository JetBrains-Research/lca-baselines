 ```python
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from fealpy.mesh import MeshFactory2d
from fealpy.pde import ADI_2d
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d, ScaledMonomialSpace2d
from fealpy.timeintegrator import ExplicitEuler
from fealpy.tools import errornorm

parser = argparse.ArgumentParser()
parser.add_argument('-nt', type=int, default=10, help='number of time divisions')
parser.add_argument('-ns', type=int, default=4, help='initial spatial division')
parser.add_argument('-ni', type=int, default=3, help='number of spatial iterations')
args = parser.parse_args()

mesh_factory = MeshFactory2d()
mesh = mesh_factory.create_mesh_gmsh('square_2d.msh')

param = ADI_2d()
param.tau = 0.01
param.sigma = 1.0
param.epsilon = 1.0

Ve = FirstKindNedelecFiniteElementSpace2d(mesh, r=1, kind='p')
Vh = ScaledMonomialSpace2d(mesh, r=1, kind='q')

def get_phi_curl_matrix(Ve):
    phi_curl = np.zeros((Ve.dof, Ve.dof))
    for i in range(Ve.ndof):
        for j in range(Ve.ndof):
            d = Ve.dof[i] - Ve.dof[j]
            if d == 1 or d == -Ve.ndof + 1:
                phi_curl[i, j] = 1
    return phi_curl

M = Ve.mass_matrix()
C = get_phi_curl_matrix(Ve)
C = np.dot(C, Ve.inverse_mass_matrix())

time = np.linspace(0, 1, args.nt + 1)
exact_u = np.sin(np.pi * time)
exact_v = np.sin(np.sqrt(2) * np.pi * time)

E = np.zeros((args.nt + 1, Ve.ndof))
H = np.zeros((args.nt + 1, Vh.ndof))
E[0, :] = Ve.interpolate(exact_u, exact_v)
H[0, :] = Vh.interpolate(exact_u, exact_v)

error_e = np.zeros(args.ni)
error_h = np.zeros(args.ni)

for iter in range(args.ni):
    print(f'spatial iteration {iter + 1}')
    integrator = ExplicitEuler(M, C, param.tau)
    for n in range(args.nt):
        b = integrator.get_right_hand_side(E[n, :], H[n, :])
        E[n + 1, :] = integrator.solve(b)
        b = integrator.get_right_hand_side(H[n, :], E[n + 1, :])
        H[n + 1, :] = integrator.solve(b)

    error_e[iter] = errornorm(Ve, E[args.nt, :], exact_u, exact_v)
    error_h[iter] = errornorm(Vh, H[args.nt, :], exact_u, exact_v)

    if iter < args.ni - 1:
        mesh.uniform_refine()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].loglog(param.h, error_e, label='error e')
axs[0].loglog(param.h, error_h, label='error h')
axs[0].set_xlabel('h')
axs[0].set_ylabel('error')
axs[0].legend()
axs[0].set_title('Error rates')

axs[1].semilogy(param.h, error_e / param.h**2, label='error e')
axs[1].semilogy(param.h, error_h / param.h**2, label='error h')
axs[1].set_xlabel('h')
axs[1].set_ylabel('error / h^2')
axs[1].legend()
axs[1].set_title('Error rates (h^2)')

plt.tight_layout()
plt.show()
```