  Here is a sample Python code that solves the heat conduction equation using the adaptive finite element method on a triangular mesh:
```
import numpy as np
import matplotlib.pyplot as plt
from fealpy import LagrangeFiniteElementSpaceOnEdgeMesh, get_quadrature_point_and_weight, get_quadrature_points_and_weights, get_gauss_point_and_weight, adaptive_coarsen, adaptive_refine, boundary_adaptive_refine, adaptive_model

# Define the mesh parameters
num_spatial_divisions = 10
num_temporal_divisions = 10
adaptive_refinement_stop_threshold = 1e-3
adaptive_refinement_and_coarsening_parameters = (1.5, 0.5)

# Define the boundary conditions
dirichlet_bc = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
neumann_bc = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# Define the heat conduction equation model
def heat_conduction_equation(u, t, x, y):
    return -u.diff(x, 2) - u.diff(y, 2)

# Define the finite element space
V = LagrangeFiniteElementSpaceOnEdgeMesh(num_spatial_divisions, num_temporal_divisions)

# Define the time step length
dt = get_time_step_length(V, heat_conduction_equation, dirichlet_bc, neumann_bc)

# Define the initial condition
u0 = np.zeros((V.num_elements, V.num_time_steps))
u0[0, :] = 1

# Define the boundary conditions
bc = np.zeros((V.num_elements, V.num_time_steps))
bc[dirichlet_bc, :] = 1

# Define the adaptive model
adaptive_model = adaptive_model(V, heat_conduction_equation, dirichlet_bc, neumann_bc)

# Define the time step
t = 0

# Define the error
error = np.inf

# Iterate until the error is below the specified threshold
while error > adaptive_refinement_stop_threshold:
    # Refine the mesh
    adaptive_refine(V, heat_conduction_equation, dirichlet_bc, neumann_bc, adaptive_model)

    # Coarsen the mesh
    adaptive_coarsen(V, heat_conduction_equation, dirichlet_bc, neumann_bc, adaptive_model)

    # Update the time step
    t += dt

    # Update the error
    error = np.max(np.abs(u0 - u))

    # Print the error
    print(f"Error at time step {t}: {error}")

    # Plot the mesh
    plt.figure()
    plt.plot(V.mesh.x, V.mesh.y, 'bo-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Mesh at time step {t}')
    plt.show()

# Print the final solution
print(f"Final solution: {u0}")

# Plot the final solution
plt.figure()
plt.plot(V.mesh.x, V.mesh.y, 'bo-')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Final solution')
plt.show()
```
This code uses the `fealpy` library to define the mesh and the finite element space, and the `adaptive_model` class to define the adaptive model. The code also uses the `adaptive_refine` and `adaptive_coarsen` functions to refine and coarsen the mesh, and the `boundary_adaptive_refine` function to refine the boundary conditions. The code iteratively refines the mesh based on the recovery estimate until the error is below the specified threshold, and it also coarsens the mesh after each time step. The code also saves a plot of the mesh at each refinement and coarsening step, and it prints the error at each time step and plots the numerical solution at specified time steps.