 Here is a simplified version of the Python code using Fealpy library for the adaptive finite element method to solve the 2D heat conduction equation. Please note that this is a basic outline and might need adjustments to fit your specific requirements.

```python
from fealpy import *
import matplotlib.pyplot as plt

def create_mesh(nx, ny, refine_params):
    mesh = Triangulation(nx, ny, refine_params=refine_params)
    return mesh

def create_function_space(mesh):
    V = FunctionSpace(mesh, "Lagrange", 1)
    return V

def create_boundary_conditions(V, boundary_conditions):
    u_dirichlet = Function(V)
    for i, bc in enumerate(boundary_conditions):
        u_dirichlet.vector()[bc[0]] = bc[1]
    return u_dirichlet

def create_problem(V, u_dirichlet):
    T = 1.0
    dt = 0.01
    alpha = 1.0
    f = Constant(0.0)
    A = Derivative(2)
    L = inner(A, u) - alpha * inner(u, u) + f
    problem = NonlinearVariationalProblem(L, u_dirichlet, V)
    return problem

def solve_problem(problem):
    solver = NonlinearVariationalSolver(problem)
    solver.solve()
    return solver.solution()

def refine_mesh(mesh, error, threshold):
    if error > threshold:
        mesh.refine(markers=mesh.get_elements_with_hmax(error))
    return mesh

def coarsen_mesh(mesh):
    mesh.coarsen()
    return mesh

def main():
    nx = 10
    ny = 10
    refine_params = RefineParameters(markers=["hmax"], threshold=0.1)
    adaptive_refine_params = AdaptiveRefineParameters(refine_threshold=0.05, coarsen_every_n_steps=10)
    mesh = create_mesh(nx, ny, refine_params)
    V = create_function_space(mesh)
    boundary_conditions = [(0, 0), (mesh.num_vertices() - 1, 1.0)]
    u_dirichlet = create_boundary_conditions(V, boundary_conditions)
    problem = create_problem(V, u_dirichlet)
    solver = NonlinearVariationalSolver(problem)
    solver.options["newton"]["max_iter"] = 10
    solver.options["newton"]["tol"] = 1e-8
    solver.options["output_iter"] = 1
    solver.solve(adaptive_refine_params)

    u = solver.solution()
    times = solver.get_times()
    errors = []
    plots = []

    for i, t in enumerate(times):
        error = compute_error(u, t)
        errors.append(error)
        if i % 5 == 0:
            plt.plot(errors)
            plt.show()
        if i != 0:
            mesh = coarsen_mesh(mesh)
        plot_mesh(mesh)
        plot_solution(u, mesh)

if __name__ == "__main__":
    main()
```

This code creates a mesh, defines the function space, creates boundary conditions, sets up the problem, solves it using the adaptive finite element method, refines and coarsens the mesh, and plots the mesh and solution at each step. The `compute_error` function and the plotting functions (`plot_mesh`, `plot_solution`) are not provided in this code snippet, as they would depend on your specific requirements and the libraries you are using for plotting.