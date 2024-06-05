```python
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import meshio

# Parse command line arguments
parser = argparse.ArgumentParser(description='Solve a Poisson equation using VEM.')
parser.add_argument('--degree', type=int, help='Degree of the VEM space')
parser.add_argument('--max_iter', type=int, help='Maximum number of iterations for mesh refinement')
parser.add_argument('--adaptive_param', type=float, help='Adaptive parameter for mesh refinement')
args = parser.parse_args()

# Initialize problem specifics
degree = args.degree
max_iter = args.max_iter
adaptive_param = args.adaptive_param
domain = ...  # Define your domain here
pde = ...  # Define your PDE here
error_matrix = []
mesh = ...  # Initialize your mesh here

# Function to set up VEM space, assemble matrices, and solve
def solve_vem(mesh, degree):
    # Placeholder for actual VEM setup and solution
    stiffness_matrix = csr_matrix(...)  # Assemble stiffness matrix
    rhs = np.array(...)  # Assemble right-hand side
    solution = spsolve(stiffness_matrix, rhs)
    return solution

# Main loop for adaptive mesh refinement
for iteration in range(max_iter):
    # Set up VEM space and function
    vem_space = ...  # Define VEM space
    
    # Assemble stiffness matrix and right-hand side
    stiffness_matrix, rhs = ...  # Assemble here
    
    # Apply Dirichlet boundary conditions
    # This is problem-specific and may involve modifying stiffness_matrix and rhs
    
    # Solve the linear system
    solution = solve_vem(mesh, degree)
    
    # Compute error and store in error_matrix
    error = ...  # Compute error here
    error_matrix.append(error)
    
    # Mark cells for refinement and refine mesh
    mesh = ...  # Refine mesh based on error/residual estimate
    
    # Check for termination condition
    if len(vem_space) > some_threshold:  # Define your threshold
        break

# Post-processing
# Display error rates
plt.figure()
plt.plot(error_matrix)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error rates')
plt.show()

# Save degrees of freedom and error matrix to text files
np.savetxt('degrees_of_freedom.txt', [len(vem_space)])
np.savetxt('error_matrix.txt', error_matrix)

# Plot final mesh
meshio.write("final_mesh.vtk", mesh)  # Assuming mesh is compatible with meshio
plt.figure()
plt.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.cells)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Mesh')
plt.show()
```