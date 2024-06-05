  import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import sparse
from fealpy import ADI_2d, FirstKindNedelecFiniteElementSpace2d, ScaledMonomialSpace2d

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--time_division', type=int, default=100)
parser.add_argument('--initial_spatial_division', type=int, default=10)
parser.add_argument('--num_spatial_iterations', type=int, default=10)
args = parser.parse_args()

# Create initial 2D triangular mesh
mesh = fealpy.Mesh2d(args.initial_spatial_division, args.initial_spatial_division)

# Define parameters for the PDE
adi = ADI_2d(mesh, args.time_division)

# Initialize electric and magnetic fields
V = FirstKindNedelecFiniteElementSpace2d(mesh, 1)
B = ScaledMonomialSpace2d(mesh, 1)

# Define function to get phi curl matrix
def get_phi_curl_matrix(V, B):
    phi_curl_matrix = sparse.dok_matrix((V.dim, B.dim), dtype=np.complex128)
    for i in range(V.dim):
        for j in range(B.dim):
            phi_curl_matrix[i, j] = V.curl(i, j) * B.curl(i, j)
    return phi_curl_matrix

# Create mass and curl matrices
mass_matrix = adi.get_mass_matrix(V)
curl_matrix = get_phi_curl_matrix(V, B)

# Iterate over number of spatial iterations
for i in range(args.num_spatial_iterations):
    # Compute right-hand side of equation for next time layer
    rhs = adi.get_rhs(V, B, mass_matrix, curl_matrix)

    # Handle boundary conditions for next time layer
    rhs = adi.apply_boundary_conditions(rhs)

    # Compute electric and magnetic fields for next time layer
    V.update(rhs)
    B.update(rhs)

    # Calculate error between computed and actual solutions
    error = np.abs(V.solution - V.exact_solution)
    error_rate = np.linalg.norm(error, np.inf) / np.linalg.norm(V.exact_solution, np.inf)
    print(f'Error rate at iteration {i+1}: {error_rate}')

    # Refine mesh uniformly if not last iteration
    if i < args.num_spatial_iterations - 1:
        mesh.refine()

# Display error matrix and plot error rates
error_matrix = np.abs(V.solution - V.exact_solution)
error_rates = np.linalg.norm(error_matrix, np.inf) / np.linalg.norm(V.exact_solution, np.inf)
plt.plot(error_rates)
plt.xlabel('Iteration')
plt.ylabel('Error rate')
plt.show()