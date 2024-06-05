 Here is a simplified Python code using Fealpy library for the heat conduction equation with adaptive finite element method. Please note that this is a basic example and might need adjustments to fit your specific needs.

```python
from fealpy.meshes import IntervalMesh, TetrahedronMesh, WedgeMesh
from fealpy.finiteelements import LagrangeFiniteElementSpaceOnEdgeMesh, ParametricLagrangeFiniteElementSpaceOnWedgeMesh
from fealpy.numerics import adaptive_refine, adaptive_coarsen, boundary_adaptive_refine
from fealpy.materials import LinearThermalMaterial
from fealpy.numerics.time_integration import BackwardEuler
from fealpy.numerics.linear_solvers import LinearSolverUMFPACK
from fealpy.numerics.quadrature import GaussQuadrature
from fealpy.plots import plot_mesh, plot_function
from fealpy.function_spaces import FunctionSpace
import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation(n_spatial_divisions, n_temporal_divisions, threshold, alpha, beta):
    # Create initial mesh
    n_spatial_divisions_x = n_spatial_divisions
    n_spatial_divisions_y = n_spatial_divisions
    mesh_x = IntervalMesh(n_spatial_divisions_x)
    mesh_y = IntervalMesh(n_spatial_divisions_y)
    mesh = WedgeMesh([mesh_x, mesh_y])

    # Define finite element space
    Vh = ParametricLagrangeFiniteElementSpaceOnWedgeMesh(mesh, degree=1)

    # Define material properties and initial condition
    material = LinearThermalMaterial(alpha=alpha, beta=beta)
    u0 = FunctionSpace.zeros(Vh)

    # Define boundary conditions
    def bc_left(x, t):
        return 1.0

    def bc_right(x, t):
        return 0.0

    def bc_bottom(x, t):
        return 0.0

    def bc_top(x, t):
        return 0.0

    boundary_conditions = {
        (mesh.boundary_markers[0], 0): bc_left,
        (mesh.boundary_markers[1], 0): bc_right,
        (mesh.boundary_markers[2], 0): bc_bottom,
        (mesh.boundary_markers[3], 0): bc_top,
    }

    # Define time-stepping scheme
    time_mesh = time_mesh(n_temporal_divisions)
    time_step_length = time_mesh.time_step_length
    time_step_lengths = time_mesh.time_step_lengths
    time_step_indices = time_mesh.time_step_indices
    time_step_lengths[-1] = np.inf

    linear_solver = LinearSolverUMFPACK()
    time_integrator = BackwardEuler(linear_solver=linear_solver)

    # Adaptive loop
    errors = []
    refined_meshes = []
    coarsened_meshes = []
    for i in range(n_temporal_divisions):
        print(f"Time step {i+1}/{n_temporal_divisions}")
        u = u0.copy()
        for j in range(time_step_indices[i], time_step_indices[i+1]):
            # Solve the heat equation
            u_new = u.copy()
            for q, w in GaussQuadrature(Vh.dof_per_cell, 1):
                x, y = q
                t = time_mesh.time(q)
                F = 0.0
                for cell in mesh.active_cells():
                    if cell.markers[0] == 0:
                        bc = boundary_conditions[(cell.markers[1], j)](x, t)
                        u_new[cell.dof_indices] += bc * w
                    else:
                        F += (material.diffusion_coefficient(x, y, t) * (u[cell.dof_indices] - u_new[cell.dof_indices])) * w
                F += material.source_term(x, y, t) * w
            time_integrator.solve(F, u_new, u)

        # Refine the mesh
        u_error = u - u0
        error = np.l2_norm(u_error.values, axis=0)
        errors.append(error)
        if error > threshold:
            refined_meshes.append(mesh)
            mesh = boundary_adaptive_refine(mesh, u_error, adaptive_refine, adaptive_coarsen)
        else:
            coarsened_meshes.append(mesh)
            mesh = adaptive_coarsen(mesh)

        # Plot the mesh and solution
        if i in [0, n_temporal_divisions//2, n_temporal_divisions-1]:
            plot_mesh(mesh)
            plt.show()
            plot_function(u, mesh, title=f"Time step {i+1}")
            plt.show()

    return errors, refined_meshes, coarsened_meshes

# Call the function
errors, refined_meshes, coarsened_meshes = solve_heat_equation(n_spatial_divisions=100, n_temporal_divisions=100, threshold=1e-4, alpha=1.0, beta=0.0)
```

This code defines a function `solve_heat_equation` that takes as input the number of spatial and temporal divisions, the adaptive refinement stop threshold, and the adaptive refinement and coarsening parameters. It uses the 2D heat conduction equation model from the fealpy library with the Lagrange finite element space. The code iteratively refines the mesh based on the recovery estimate until the error is below the specified threshold. The code also coarsens the mesh after each time step, plots the mesh at each refinement and coarsening step, and prints the error at each time step. The numerical solution is plotted at specified time steps.