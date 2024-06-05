 Here is a simplified version of the Python code using the `FEniCS` library for solving the Poisson equation using the Virtual Element Method (VEM) on a polygonal mesh. This code does not include the specific PDE and domain setup, as it would require more context and details.

```python
import argparse
import numpy as np
import fenics as fe

parser = argparse.ArgumentParser()
parser.add_argument('--degree', type=int, default=1)
parser.add_argument('--max_iter', type=int, default=10)
parser.add_argument('--max_dofs', type=int, default=100000)
parser.add_argument('--adaptive_param', type=float, default=0.25)
args = parser.parse_args()

# Initialize problem, domain, and mesh
mesh = fe.Mesh("polygon_mesh.xml")
V = fe.VectorElement("P", mesh.ufl_cell(), degree=args.degree)

# Define function space and set up function
Vm = fe.FunctionSpace(mesh, V)
u = fe.Function(Vm)

# Define error matrix and initialize with zeros
error_matrix = fe.Matrix()
error_matrix.resize((args.max_iter, args.max_iter))

for iter in range(args.max_iter):
    # Set up VEM space and assemble stiffness matrix and right-hand side
    A, b = setup_linear_system(Vm, u)

    # Apply Dirichlet boundary conditions
    apply_boundary_conditions(u)

    # Solve the linear system
    u.solve(A, b)

    # Compute error and mark cells for refinement
    error, markers = compute_error_and_refinement_markers(u)

    # Adaptively refine the mesh
    mesh = fe.Mesh(mesh, markers, refine_number=2)

    # Update error matrix
    error_matrix[iter, iter] = error

    # Check if the number of degrees of freedom exceeds the threshold
    if fe.Mesh.num_dofs(mesh) > args.max_dofs:
        break

# Display error rates, save number of degrees of freedom and error matrix to text files
print("Error rates:")
print(error_matrix)
save_dofs_and_error(mesh, error_matrix, "dofs_and_error.txt")

# Plot error
plot_error(error)

# Plot the final mesh
plot_mesh(mesh)
```

This code assumes that you have a polygon mesh file named `polygon_mesh.xml` and that the functions `setup_linear_system`, `apply_boundary_conditions`, `compute_error_and_refinement_markers`, `save_dofs_and_error`, `plot_error`, and `plot_mesh` are defined elsewhere in your code. You will need to implement these functions according to your specific problem setup and requirements.