 ```python
import argparse
import dolfinx
import dolfinx.mesh
import dolfinx.fem
import dolfinx.logging
import ufl
import numpy as np
import matplotlib.pyplot as plt

dolfinx.logging.set_log_level(dolfinx.logging.LogLevel.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--degree", type=int, default=1, help="degree of VEM space")
parser.add_argument("--max_iter", type=int, default=5, help="maximum number of iterations for mesh refinement")
parser.add_argument("--adaptive_parameter", type=float, default=0.5, help="adaptive parameter")
args = parser.parse_args()

domain = (0, 1, 0, 1)
pde = ufl.div(ufl.grad(u))

mesh = dolfinx.mesh.create_unit_square(dolfinx.mesh.GhostMode.shared_facet, 2, 2)
V = dolfinx.fem.VectorFunctionSpace(mesh, ("DG", args.degree))

error_matrix = np.zeros((args.max_iter,))
dofs = V.dofmap.index_map.size_local

for n in range(args.max_iter):
    u = dolfinx.fem.Function(V)
    a = dolfinx.fem.form(pde*ufl.dx)
    L = dolfinx.fem.form(ufl.Constant(0)*ufl.dx)

    bc = dolfinx.fem.dirichlet_bc(V, ufl.Constant(0), lambda x: (x[0] < 1e-14) | (x[0] > 1 - 1e-14))

    b = dolfinx.fem.petsc.assemble_right_hand_side(L, bc, dolfinx.fem.petsc.MatType.aij, V.dofmap)
    A = dolfinx.fem.petsc.assemble_matrix(a, bc, dolfinx.fem.petsc.MatType.aij, V.dofmap, V.dofmap)

    dolfinx.fem.petsc.set_bc(A, bc)
    dolfinx.fem.petsc.set_bc(b, bc)

    u.x.set(np.zeros(dofs))
    dolfinx.fem.petsc.solve(A, u.x, b)

    error_local = np.zeros(dofs)
    dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(ufl.TestFunction(V))) * ufl.dx).apply(V.dofmap.restriction, V.dofmap.local_range, error_local)
    error = np.sqrt(np.dot(error_local, error_local))
    error_matrix[n] = error

    if n < args.max_iter - 1:
        marked_cells = dolfinx.mesh.refine(mesh, error_local, args.adaptive_parameter)
        mesh = dolfinx.mesh.create_mesh(2, marked_cells, mesh.topology.dim)

print("Error rates:")
for i in range(args.max_iter):
    print(f"Iteration {i}: {error_matrix[i]:.4e}")

np.savetxt("dofs.txt", [dofs])
np.savetxt("error_matrix.txt", error_matrix)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].semilogy(error_matrix)
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Error")
axs[0].set_title("Error convergence")
axs[1].triplot(mesh, linewidth=0.2)
axs[1].set_title("Final mesh")
plt.tight_layout()
plt.savefig("errors_and_mesh.png")
plt.show()
```