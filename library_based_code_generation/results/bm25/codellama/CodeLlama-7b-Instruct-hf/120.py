  import sys
from fealpy import *

# Get command line arguments
max_iterations = int(sys.argv[1])
theta = float(sys.argv[2])
k = float(sys.argv[3])

# Initialize mesh
mesh = HalfEdgeMesh()

# Define variables
A = matrix_A(mesh, theta, k)
b = a_u(mesh, theta, k)
x = np.zeros(mesh.num_dofs)

# Create loop for maximum number of iterations
for i in range(max_iterations):
    # Solve problem
    x = np.linalg.solve(A, b)

    # Calculate residuals and high order terms
    residuals = np.linalg.norm(b - A @ x)
    high_order_terms = np.linalg.norm(x)

    # Save results and error data to file
    save_data(mesh, x, residuals, high_order_terms, i)

    # Plot mesh and save as image file
    mesh.plot(save_as_image=True)

    # Refine mesh based on residuals
    if i < max_iterations - 1:
        mesh.refine(residuals)

# Save final error data to file
save_data(mesh, x, residuals, high_order_terms, max_iterations)

# Display multi-rate plot
plot_multi_rate(mesh, x, residuals, high_order_terms)