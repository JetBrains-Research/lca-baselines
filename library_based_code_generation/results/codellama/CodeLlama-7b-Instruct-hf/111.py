  import numpy as np
import simsopt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.constants import mu_0
from scipy.spatial import Delaunay
import vtk

# Define the objective function
def objective(x):
    # Initialize the boundary magnetic surface
    B_surf = np.zeros((N, 3))
    B_surf[:, 0] = x[0::3]
    B_surf[:, 1] = x[1::3]
    B_surf[:, 2] = x[2::3]

    # Create equally spaced curves and multifilament grid
    curves = np.zeros((N, 3))
    curves[:, 0] = x[0::3]
    curves[:, 1] = x[1::3]
    curves[:, 2] = x[2::3]
    grid = np.zeros((N, 3))
    grid[:, 0] = x[0::3]
    grid[:, 1] = x[1::3]
    grid[:, 2] = x[2::3]

    # Apply stellarator and rotation symmetries
    B_surf = np.dot(B_surf, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
    curves = np.dot(curves, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
    grid = np.dot(grid, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))

    # Define the Biot-Savart law
    def B_field(r, theta, phi):
        B_r = np.zeros(r.shape)
        B_theta = np.zeros(r.shape)
        B_phi = np.zeros(r.shape)
        for i in range(N):
            B_r += (mu_0/4/np.pi)*(3*np.cos(theta[i])*np.cos(phi[i])*B_surf[i, 0] - np.sin(theta[i])*B_surf[i, 1])
            B_theta += (mu_0/4/np.pi)*(3*np.cos(theta[i])*np.sin(phi[i])*B_surf[i, 0] + np.cos(theta[i])*B_surf[i, 1])
            B_phi += (mu_0/4/np.pi)*(np.sin(theta[i])*B_surf[i, 2])
        return B_r, B_theta, B_phi

    # Compute the squared flux
    F = np.zeros(N)
    for i in range(N):
        r, theta, phi = np.meshgrid(grid[:, 0], grid[:, 1], grid[:, 2], indexing='ij')
        B_r, B_theta, B_phi = B_field(r, theta, phi)
        F[i] = np.trapz(np.trapz(np.trapz(B_r*B_r + B_theta*B_theta + B_phi*B_phi, grid[:, 1]), grid[:, 2]), grid[:, 0])

    # Compute the curve length penalty
    L = np.zeros(N)
    for i in range(N):
        L[i] = np.sum(np.sqrt(np.sum((curves[i, :] - curves[i+1:, :])**2, axis=1)))

    # Compute the coil-to-coil distance penalty
    D = np.zeros(N)
    for i in range(N-1):
        D[i] = np.sqrt(np.sum((grid[i, :] - grid[i+1, :])**2))

    # Compute the total objective function
    obj = F + Lambda*L + Delta*D

    # Perform a Taylor test
    if np.any(np.isnan(obj)):
        return np.inf

    # Run the optimization using L-BFGS-B method from scipy.optimize
    res = minimize(obj, x, method='L-BFGS-B', jac=True, options={'disp': False})

    # Save the results in VTK format
    vtkfile = vtk.vtkUnstructuredGrid()
    vtkfile.SetPoints(vtk.vtkPoints())
    for i in range(N):
        vtkfile.GetPoints().InsertNextPoint(grid[i, :])
    vtkfile.SetCells(vtk.VTK_LINE, np.array([[i, i+1] for i in range(N-1)]))
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName('output.vtu')
    writer.SetInputData(vtkfile)
    writer.Write()

# Define the parameters
N = 100
Lambda = 1e-3
Delta = 1e-3
x = np.zeros(3*N)

# Perform the optimization
objective(x)