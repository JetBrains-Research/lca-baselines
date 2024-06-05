 Here is a simplified Python code using Fealpy library for the given problem. Please note that this is a basic example and might need adjustments based on your specific requirements.

```python
import argparse
import fealpy as fp
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--theta', type=float, default=0.5)
parser.add_argument('--k', type=int, default=3)
args = parser.parse_args()

max_iter = args.max_iter
theta = args.theta
k = args.k

# Define the domain and mesh
domain = fp.MeshDomain.rectangle(0, 1, 0, 1)
mesh = fp.Mesh.generate_halfedge_mesh(domain, n=100, theta=theta)

# Define the problem
problem = fp.FrictionProblem(mesh, kappa=theta)

# Define the solution and error data lists
solutions = []
errors = []

for i in range(max_iter):
    # Solve the problem
    solution = problem.solve()

    # Calculate residuals and high order terms
    residual = problem.residual(solution)
    high_order_terms = problem.high_order_terms(solution)

    # Save the results and error data to a file
    solutions.append(solution)
    errors.append(high_order_terms)
    np.savetxt(f"solution_{i}.txt", solution.vector.get_data())
    np.savetxt(f"error_{i}.txt", high_order_terms.vector.get_data())

    # Refine the mesh based on the residuals
    if i < max_iter - 1:
        problem.refine_mesh(residual > 1e-6)

    # Plot the mesh and save it as an image file
    plt.figure()
    mesh.plot_triangles()
    plt.savefig(f"mesh_{i}.png")
    plt.clf()

# Save the final error data to a file
np.savetxt("final_error.txt", errors[-1].vector.get_data())

# Display a multi-rate plot
problem.plot_multi_rate(solutions)
plt.show()
```

This code assumes that you have Fealpy and Matplotlib installed. If not, you can install them using pip:

```
pip install fealpy matplotlib
```