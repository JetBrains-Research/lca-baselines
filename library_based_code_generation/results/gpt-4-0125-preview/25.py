```python
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux, CoilLength, MinimumDistance, Curvature, SquaredCurvature
from simsopt.optimize.objectives import LeastSquaresProblem
from simsopt.optimize.dogleg import Dogleg
from simsopt._core.graph_optimizable import Optimizable

# Initialize the boundary magnetic surface
nfp = 1  # Number of field periods
nphi = 16
nzeta = 16
surface = SurfaceRZFourier(nphi=nphi, nzeta=nzeta, nfp=nfp, mpol=3, ntor=3)
surface.set_coeffs(np.random.randn(*surface.get_coeffs().shape))

# Create initial coils
num_coils = 4
coils = []
currents = []
for i in range(num_coils):
    coil = np.array([np.cos(2*np.pi*i/num_coils), np.sin(2*np.pi*i/num_coils), 0, 0.1, 0.1, 0.1, 0.1])
    coils.append(coil)
    currents.append(Current(1000))
coils = coils_via_symmetries(coils, currents, nfp, True)

# Define Biot-Savart
bs = BiotSavart(coils)
bs.set_points(surface.gamma().reshape((-1, 3)))

# Define the individual terms of the objective function
target_field = np.zeros(surface.gamma().shape[:-1])
flux_objective = SquaredFlux(surface, bs, target_field)
coil_length = CoilLength(coils)
min_distance = MinimumDistance(coils, 0.1)
curvature = Curvature(coils)
mean_squared_curvature = SquaredCurvature(coils)

# Form the total objective function
total_objective = LeastSquaresProblem([flux_objective, coil_length, min_distance, curvature, mean_squared_curvature])

# Perform a Taylor test
Optimizable.taylor_test(total_objective, verbose=True)

# Run the optimization
solver = Dogleg(maxiter=20)
x0 = total_objective.x0()
result = solver.solve(total_objective, x0)

# Use the result as the initial guess for a subsequent optimization with reduced penalty for the coil length
coil_length.weight = 0.1
result2 = solver.solve(total_objective, result.x)

# Save the optimized coil shapes and currents
for i, coil in enumerate(coils):
    np.savetxt(f"coil_{i}_shape_optimized.txt", coil.x)
    np.savetxt(f"coil_{i}_current_optimized.txt", [coil.current.value])
```