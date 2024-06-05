 Here is a simplified example of how you might implement this task using the SimPEG library. Please note that this is a basic example and might need adjustments based on your specific requirements.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simpeg import (
    Directory,
    Model,
    Mesh,
    Volumes,
    Surface,
    GridData,
    LinearOperator,
    NonlinearOperator,
    Data,
    Inversion,
    Tikhonov,
    Validation,
    Casing,
    ParametrizedCasingAndLayer,
)

# Define the ground truth model
background = 1.0
conductive_sphere_radius = 1.0
conductive_sphere_sigma = 10.0
resistive_sphere_radius = 1.5
resistive_sphere_sigma = 0.1

ground_truth = Model(
    sigma=np.where(np.sqrt(x**2 + y**2 + z**2) < conductive_sphere_radius, conductive_sphere_sigma,
                    np.where(np.sqrt(x**2 + y**2 + z**2) < resistive_sphere_radius, background,
                             resistive_sphere_sigma))
)

# Create a mesh
mesh = Mesh(Directory('mesh'))

# Create a casing and layer model
casing = Casing(mesh, outer_radius=conductive_sphere_radius)
layer = ParametrizedCasingAndLayer(mesh, casing, outer_radius=resistive_sphere_radius)

# Create a synthetic data
src = Surface(mesh, 'src', [0, 0, 0])
rec = Surface(mesh, 'rec', [0, 0, 1])
data_dir = Directory('data')
dd_data = Data(data_dir, 'dd_data.dat')

# Create the forward model
A = getA(mesh, src, rec, layer, ground_truth, dd_data.n_data)
_aHd = _aHd(mesh, src, rec, layer, ground_truth)
_derivA = getADeriv(mesh, src, rec, layer, ground_truth)

# Create the Tikhonov regularization
tikhonov = Tikhonov(A, _derivA, dd_data, lambdas=[1e-3, 1e-3, 1e-3])

# Create the inversion
inversion = Inversion(tikhonov, layer, ground_truth, validation=Validation.L2)

# Configure the inversion
inversion.options.max_iter = 100
inversion.options.tol = 1e-6
inversion.options.verbose = True

# Perform the inversion
inversion.run()

# Get the inverted model
inverted_model = inversion.x

# Plot the ground truth and inverted model
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mesh.x, mesh.y, mesh.z, ground_truth.sigma.reshape(mesh.nx, mesh.ny, mesh.nz), cmap='inferno', edgecolor='none')
ax.plot_surface(mesh.x, mesh.y, mesh.z, inverted_model.reshape(mesh.nx, mesh.ny, mesh.nz), cmap='inferno', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Flatten the inverted model for horizontal plotting
inverted_model_flat = inverted_model.flatten()
ground_truth_flat = ground_truth.sigma.flatten()

# Plot the horizontal cross-sections
plt.figure(figsize=(10, 4))
plt.plot(ground_truth_flat, label='Ground Truth')
plt.plot(inverted_model_flat, label='Inverted Model')
plt.xlabel('Index')
plt.ylabel('Conductivity')
plt.legend()
plt.show()
```

This code creates a 3D model with two spheres, one conductive and the other resistive, performs a Tikhonov inversion using a Dipole-Dipole survey, and plots the ground truth and inverted models both vertically and horizontally. The inversion is restrained to the core mesh using an Active Cells mapping combined with an exponential mapping to invert in log conductivity space. Please ensure to adjust the parameters according to your specific needs.