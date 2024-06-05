 import dolfinx
import dolfinx.mesh
import dolfinx.fem
import dolfinx.logging
import ufl
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

degree = int(sys.argv[1])
max_iter = int(sys.argv[2])
adaptive_parameter = float(sys.argv[3])

problem = ...  # Initialize the problem with a specific PDE and domain
error_matrix = ...  # Initialize the error matrix
mesh = ...  # Initialize the mesh

dofs = dolfinx.fem.count_dofs(problem.function_space)
print(f"Initial number of degrees of freedom: {dofs}")

for n in range(max_iter):
    V = dolfinx.fem.FunctionSpace(mesh, ("DG", degree))
    u = dolfinx.fem.Function(V)
    u_ = dolfinx.fem.Function(V)

    matrix_A = dolfinx.fem.petsc.Matrix(V.dofmap.index_map, V.dofmap.index_map)
    vector_b = dolfinx.fem.petsc.Vector(V.dofmap.index_map)

    assembly_for_sspace_and_vspace_with_vector_basis(matrix_A, vector_b, problem.sspace, problem.vspace, problem.a, V, V, ufl.TestFunction(V))

    test_assembly_cell_righthand_side_and_matrix(matrix_A, vector_b, problem.sspace, problem.vspace, problem.f, V, V, ufl.TestFunction(V))

    bc = dolfinx.fem.dirichlet_bc(V, problem.bcs)
    dolfinx.fem.apply_boundary_condition(bc, vector_b, u)

    solver = dolfinx.fem.petsc.LinearSolver(matrix_A, "mumps")
    solver.solve(u.vector, vector_b)

    error = np.abs(u - u_).max()
    error_matrix[n] = error

    if n < max_iter - 1:
        marked_cells = dolfinx.mesh.refine.mark_cells(mesh, problem.sspace, problem.vspace, problem.a, u, adaptive_parameter)
        mesh = dolfinx.mesh.refine.refine(mesh, marked_cells)

    dofs = dolfinx.fem.count_dofs(V)
    if dofs > 1e6:
        break

error_rates = np.log(error_matrix[1:]) / np.log(2)
print(f"Error rates: {error_rates}")

np.savetxt("error_matrix.txt", error_matrix)
np.savetxt("dofs.txt", [dofs])

fig, ax = plt.subplots()
ax.semilogy(error_matrix, label="Error")
ax.semilogy(2**np.arange(len(error_matrix)), error_rates, label="Error rates")
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
ax.legend()
plt.savefig("error.png")

fig, ax = plt.subplots()
dolfinx.plot.plot(mesh, ax=ax)
plt.savefig("final_mesh.png")
plt.show()