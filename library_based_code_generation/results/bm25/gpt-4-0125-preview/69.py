```python
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# Assuming a hypothetical VEM library exists as vem_lib
import vem_lib

# Parse command line arguments
parser = argparse.ArgumentParser(description='Solve a Poisson equation using VEM.')
parser.add_argument('--degree', type=int, help='Degree of the VEM space')
parser.add_argument('--max_iter', type=int, help='Maximum number of iterations for mesh refinement')
parser.add_argument('--adaptive_param', type=float, help='Adaptive parameter for mesh refinement')
args = parser.parse_args()

# Initialize problem specifics (PDE, domain, etc.)
# This is highly dependent on the problem and the library's API
# Placeholder for initialization
domain = ...
pde = ...

# Initialize error matrix and mesh
error_matrix = []
mesh = vem_lib.initialize_mesh(domain)

for iteration in range(args.max_iter):
    # Set up VEM space and function
    vem_space = vem_lib.create_vem_space(mesh, args.degree)
    function = ...

    # Assemble stiffness matrix and right-hand side
    stiffness_matrix = vem_lib.assembly_for_sspace_and_vspace_with_vector_basis(vem_space, mesh)
    rhs = vem_lib.test_assembly_cell_righthand_side_and_dof_matrix(vem_space, mesh, function)

    # Apply Dirichlet boundary conditions
    bc_applied_matrix, bc_applied_rhs = vem_lib.test_dirichlet_and_neumann_bc_on_interval_mesh(stiffness_matrix, rhs, vem_space, mesh)

    # Solve the linear system
    solution = np.linalg.solve(bc_applied_matrix, bc_applied_rhs)

    # Compute error
    error = vem_lib.compute_error(solution, vem_space, mesh)
    error_matrix.append(error)

    # Mark cells for refinement and adaptively refine the mesh
    cells_to_refine = vem_lib.mark_cells_for_refinement(mesh, solution, args.adaptive_param)
    mesh = vem_lib.refine_mesh(mesh, cells_to_refine)

    # Terminate if the number of degrees of freedom exceeds a threshold
    if vem_lib.number_of_dofs(mesh) > SOME_THRESHOLD:
        print("Terminating due to exceeding the degrees of freedom threshold.")
        break

# Display error rates
plt.plot(error_matrix)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error rates')
plt.show()

# Save the number of degrees of freedom and the error matrix to text files
np.savetxt("degrees_of_freedom.txt", [vem_lib.number_of_dofs(mesh)], fmt='%d')
np.savetxt("error_matrix.txt", error_matrix, fmt='%f')

# Plot the final mesh
vem_lib.plot_mesh(mesh)
plt.title('Final Mesh')
plt.show()
```