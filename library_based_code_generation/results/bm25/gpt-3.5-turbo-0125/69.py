import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add necessary library imports for VEM
from VEM_library import *

# Parse command line arguments
parser = argparse.ArgumentParser(description='Solving Poisson equation using VEM on a polygonal mesh')
parser.add_argument('--degree', type=int, default=1, help='Degree of the VEM space')
parser.add_argument('--max_iterations', type=int, default=100, help='Maximum number of iterations for mesh refinement')
parser.add_argument('--adaptive_param', type=float, default=0.1, help='Adaptive parameter for mesh refinement')
args = parser.parse_args()

# Initialize problem with specific PDE and domain
pde = 'Poisson'
domain = 'polygonal_mesh'

# Set up error matrix and mesh
error_matrix = np.zeros(args.max_iterations)
mesh = initialize_mesh(domain)

# Enter loop for maximum number of iterations
for i in range(args.max_iterations):
    # Set up VEM space and function
    VEM_space = setup_VEM_space(degree=args.degree)
    function = initialize_function(VEM_space)

    # Assemble stiffness matrix and right-hand side
    stiffness_matrix = assemble_stiffness_matrix(VEM_space)
    rhs = assemble_rhs(VEM_space)

    # Apply Dirichlet boundary conditions
    apply_dirichlet_bc(VEM_space, stiffness_matrix, rhs)

    # Solve linear system
    solution = solve_linear_system(stiffness_matrix, rhs)

    # Compute error
    error = compute_error(VEM_space, solution)
    error_matrix[i] = error

    # Mark cells for refinement and adaptively refine mesh
    mark_cells_for_refinement(VEM_space, error, adaptive_param=args.adaptive_param)
    adaptively_refine_mesh(mesh)

    # Check termination condition
    if VEM_space.num_dofs > threshold:
        break

# Display error rates
plt.plot(error_matrix)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error Rates')
plt.show()

# Save number of degrees of freedom and error matrix to text files
np.savetxt('degrees_of_freedom.txt', VEM_space.num_dofs)
np.savetxt('error_matrix.txt', error_matrix)

# Plot error and final mesh
plot_error(error_matrix)
plot_mesh(mesh)