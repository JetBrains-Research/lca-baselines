 Here is a simplified Python code using the FEniCS library for solving a Poisson equation using the Virtual Element Method (VEM) on a polygonal mesh. Please note that this is a basic example and might need adjustments to fit your specific needs.

```python
import argparse
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--degree', type=int, default=1)
parser.add_argument('--max_iter', type=int, default=10)
parser.add_argument('--adaptive_param', type=float, default=1e-4)
parser.add_argument('--threshold', type=int, default=1e6)
args = parser.parse_args()

# Initialize problem with a specific PDE and domain
mesh = fe.UnitSquareMesh(16, 16)
u = fe.Function(fe.VectorElement('P', mesh.ufl_cell(), degree=args.degree))
v = fe.TestFunction(fe.VectorElement('P', mesh.ufl_cell(), degree=args.degree))

f = fe.Constant((0.0, 0.0))
a = fe.dot(fe.grad(u), fe.grad(v)) * dx
L = fe.dot(f, v) * dx

# Set up error matrix and mesh
error_matrix = np.zeros((args.max_iter, 2))
meshes = [mesh]

for i in range(args.max_iter):
    # Set up VEM space and the function
    sspace = fe.VectorFunctionSpace(mesh, fe.VectorElement('P', mesh.ufl_cell(), degree=args.degree))
    vspace = sspace.sub(1)

    # Assemble stiffness matrix and the right-hand side
    matrix_A = fe.assemble(a)
    b = fe.assemble(L)

    # Apply Dirichlet boundary conditions
    boundaries = fe.DirichletBC(sspace, fe.Constant((0.0, 0.0)), fe.compose(fe.Constant((0.0, 0.0)), fe.FunctionMap(fe.MeshFunction('size_t', mesh, fe.MeshEntity.vertices))) == 1)
    boundaries.apply(u)

    # Solve the linear system
    u.solve(matrix_A, b, solver_parameters={'linear_solver': 'mumps', 'ksp_type': 'preonly', 'ksp_rtol': args.adaptive_param, 'ksp_max_it': 100})

    # Compute the error
    u_exact = fe.interpolate(fe.Expression('x[0] + sin(2*pi*x[1])', degree=args.degree), sspace)
    error = fe.norm(u - u_exact) / fe.norm(u_exact)

    # Mark cells for refinement based on the residual estimate and adaptively refine the mesh
    h = fe.Function(fe.VectorElement('P', mesh.ufl_cell(), degree=args.degree))
    fe.project(fe.sqrt(fe.dot(fe.grad(u), fe.grad(u))), h)
    markers = fe.MeshFunction('size_t', mesh, fe.MeshEntity.cells)
    markers.set_all(0)
    markers.set_value(1, h > args.adaptive_param)
    mesh = fe.Mesh(mesh, markers, 'h1')

    # Update error matrix
    error_matrix[i] = [i, error]

    # Terminate if the number of degrees of freedom exceeds a certain threshold
    if fe.cell_dof_map(sspace).size() > args.threshold:
        break

# Display the error rates, save the number of degrees of freedom and the error matrix to text files, and plot the error
print(error_matrix)
np.savetxt('degrees_of_freedom.txt', [fe.cell_dof_map(sspace).size() for _ in range(args.max_iter)])
np.savetxt('error_matrix.txt', error_matrix)

# Plot the error
plt.plot(error_matrix[:, 0], error_matrix[:, 1])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()

# Plot the final mesh
fe.plot(mesh, subdomain_id=1, cmap='viridis')
plt.show()
```

This code assumes that you have FEniCS and Matplotlib installed. You can install them using:

```
pip install fenics matplotlib
```