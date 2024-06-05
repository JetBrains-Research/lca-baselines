 ```python
import fealpy
import matplotlib.pyplot as plt

# User inputs
n_space = 10  # number of spatial divisions
n_time = 100  # number of temporal divisions
tol = 1e-5  # adaptive refinement stop threshold
theta_refine = 0.3  # adaptive refinement parameter
theta_coarsen = 0.6  # adaptive coarsening parameter
save_plots = True  # save plots of the mesh at each refinement and coarsening step
plot_solution = True  # plot the numerical solution at specified time steps
plot_time_steps = [0, n_time]  # time steps to plot the numerical solution

# Define the heat conduction equation model
model = fealpy.TimeModel(fealpy.HeatEquation2D, n_space, n_time, theta_refine, theta_coarsen)

# Define the finite element space
mesh = fealpy.EdgeMesh.square_mesh(n_space, n_space)
V = fealpy.LagrangeFiniteElementSpaceOnEdgeMesh(mesh)

# Define the Dirichlet boundary conditions
def boundary_condition(x, on_boundary):
    return fealpy.test_dirichlet_and_neumann_bc_on_interval_mesh(x, on_boundary, 0)

# Set the Dirichlet boundary conditions
model.set_boundary_condition(boundary_condition)

# Initialize the solution and the error
u = fealpy.Function(V)
e = fealpy.Function(V)

# Initialize the adaptive refinement and coarsening parameters
model.set_adaptive_parameters(tol, theta_refine, theta_coarsen)

# Iteratively refine the mesh based on the recovery estimate until the error is below the specified threshold
while model.estimate > tol:
    # Refine the mesh
    mesh = model.adaptive_refine(mesh, u)

    # Save a plot of the mesh
    if save_plots:
        plt.savefig(f'mesh_{model.current_time_step}.png')

    # Compute the numerical solution
    model.solve(u)

    # Compute the error
    model.compute_error(u, e)

# Coarsen the mesh after each time step
for _ in range(model.get_number_of_time_steps()):
    mesh = model.adaptive_coarsen(mesh, u)

    # Save a plot of the mesh
    if save_plots:
        plt.savefig(f'mesh_{model.current_time_step}.png')

    # Compute the numerical solution
    model.solve(u)

    # Compute the error
    model.compute_error(u, e)

    # Print the error
    print(f'Time step {model.current_time_step}: error = {model.estimate:.4e}')

    # Plot the numerical solution
    if plot_solution and model.current_time_step in plot_time_steps:
        plt.figure()
        model.plot_solution(u)
        plt.show()
```