 ```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.uniform_mesh import UniformMesh2d
from fealpy.fem import LagrangeFiniteElementSpace2d
from fealpy.timeintegrator import BackwardEuler
from fealpy.pde.heat_equation import HeatEquation2d
from fealpy.adaptive.adaptive_refine import adaptive_refine_uniform
from fealpy.adaptive.adaptive_coarsen import adaptive_coarsen_uniform
from fealpy.visualization import savefig
from fealpy.tools.plot import plot

# Create a uniform mesh
nx = 10
ny = 10
mesh = UniformMesh2d(np.array([0, 0, 1, 0, 1, 1, 0, 1]), np.array([0, 1, 1, 0, 0, 1, 1, 1]))

# Create a finite element space
fes = LagrangeFiniteElementSpace2d(mesh, p=1)

# Create a time integrator
dt = 0.01
ti = BackwardEuler(fes)

# Create a PDE model
model = HeatEquation2d(fes, ti)

# Set the Dirichlet boundary conditions
model.set_dirichlet_bc([1, 2, 3, 4], [0, 0, 0, 0])

# Set the adaptive refinement stop threshold
threshold = 1e-5

# Set the adaptive refinement and coarsening parameters
refine_param = 0.5
coarsen_param = 0.1

# Set the time step and total time
t_end = 1
nsteps = int(t_end / dt)

# Initialize the error list
error_list = []

# Iterate over time steps
for n in range(nsteps):
    # Solve the PDE model
    uh = model.solve()

    # Calculate the error
    error = model.estimate_error(uh)
    error_list.append(error)

    # Plot the numerical solution
    if (n + 1) % 10 == 0:
        plot(model.mesh, uh, 'Numerical Solution at t=%g' % (n * dt))
        plt.show()

    # Check if the error is below the threshold
    if error < threshold:
        break

    # Refine the mesh
    mesh = adaptive_refine_uniform(model.mesh, model.error_estimator, refine_param, threshold)

    # Save a plot of the mesh
    savefig(model.mesh, 'mesh_%04d.png' % n)

# Coarsen the mesh
mesh = adaptive_coarsen_uniform(mesh, coarsen_param)

# Plot the final error
plt.loglog(error_list, 'o-')
plt.title('Error history')
plt.xlabel('Time step')
plt.ylabel('Error')
plt.show()
```
Please note that this is a basic implementation of the heat conduction equation using the adaptive finite element method on a triangular mesh. The code may need to be modified to fit the specific requirements of your problem.