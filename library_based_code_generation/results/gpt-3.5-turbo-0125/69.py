import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import necessary libraries and modules for VEM
from VEM import VEMSpace, assemble_stiffness_matrix, assemble_rhs, apply_dirichlet_bc, solve_linear_system, compute_error
from mesh import Mesh, adaptively_refine_mesh

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--degree", type=int, help="Degree of the VEM space")
parser.add_argument("--max_iterations", type=int, help="Maximum number of iterations for mesh refinement")
parser.add_argument("--adaptive_parameter", type=float, help="Adaptive parameter")
args = parser.parse_args()

# Initialize problem with specific PDE and domain
pde = "Poisson Equation"
domain = "Polygonal Mesh"

# Set up error matrix and mesh
error_matrix = np.zeros(args.max_iterations)
mesh = Mesh()

# Enter loop for maximum number of iterations
for i in range(args.max_iterations):
    # Set up VEM space and function
    vem_space = VEMSpace(args.degree)
    
    # Assemble stiffness matrix and right-hand side
    stiffness_matrix = assemble_stiffness_matrix(vem_space)
    rhs = assemble_rhs(vem_space)
    
    # Apply Dirichlet boundary conditions
    stiffness_matrix, rhs = apply_dirichlet_bc(stiffness_matrix, rhs)
    
    # Solve linear system
    solution = solve_linear_system(stiffness_matrix, rhs)
    
    # Compute error
    error = compute_error(solution)
    error_matrix[i] = error
    
    # Mark cells for refinement and adaptively refine mesh
    marked_cells = mesh.mark_cells_for_refinement(error, args.adaptive_parameter)
    mesh = adaptively_refine_mesh(mesh, marked_cells)
    
    # Check if number of degrees of freedom exceeds threshold
    if mesh.num_dofs > threshold:
        break

# Display error rates
error_rates = np.diff(np.log(error_matrix))
print("Error rates:", error_rates)

# Save number of degrees of freedom and error matrix to text files
np.savetxt("degrees_of_freedom.txt", mesh.num_dofs)
np.savetxt("error_matrix.txt", error_matrix)

# Plot error and final mesh
plt.plot(error_matrix)
plt.title("Error vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()

mesh.plot()