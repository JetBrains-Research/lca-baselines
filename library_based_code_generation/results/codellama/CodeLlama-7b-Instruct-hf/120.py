  import sys
import numpy as np
import matplotlib.pyplot as plt
from fealpy import HalfedgeMesh, SimplifiedFrictionProblem

# Get command line arguments
max_iterations = int(sys.argv[1])
theta = float(sys.argv[2])
k = float(sys.argv[3])

# Initialize mesh and problem
mesh = HalfedgeMesh.from_file("input.obj")
problem = SimplifiedFrictionProblem(mesh, theta, k)

# Create loop for maximum number of iterations
for i in range(max_iterations):
    # Solve problem, calculate residuals and high order terms
    problem.solve()
    residuals = problem.residuals()
    high_order_terms = problem.high_order_terms()

    # Save results and error data to file
    np.savetxt("results.txt", residuals, fmt="%.16e")
    np.savetxt("high_order_terms.txt", high_order_terms, fmt="%.16e")

    # Plot mesh and save as image file
    mesh.plot(show=False, save="mesh.png")

    # Refine mesh based on residuals
    if i < max_iterations - 1:
        mesh.refine(residuals)

# Save final error data to file
np.savetxt("final_error.txt", residuals, fmt="%.16e")

# Display multi-rate plot
plt.plot(residuals)
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.title("Multi-rate Plot")
plt.show()