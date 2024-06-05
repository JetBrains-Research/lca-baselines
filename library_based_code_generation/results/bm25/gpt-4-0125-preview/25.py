```python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.coil import Current, Coil
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux, CoilLength, MinimumDistance, Curvature, MeanSquaredCurvature
from simsopt.optimize.objectives import LeastSquaresProblem, least_squares_serial_solve
from simsopt.objectives.least_squares import LeastSquaresProblemLS
from simsopt.geo.curveobjectives import CurveLength, MinimumDistanceCurveCurve, MinimumDistanceCurveSurface
import numpy as np

# Initialize the boundary magnetic surface
nfp = 1  # Number of field periods
nphi = 10  # Resolution in the toroidal direction
nzeta = 10  # Resolution in the poloidal direction
surface = SurfaceRZFourier(nphi=nphi, nzeta=nzeta, nfp=nfp, stellsym=True)

# Create initial coils
num_coils = 4
coils = []
currents = []
for i in range(num_coils):
    coil = Coil.circular_coil(radius=1.0 + 0.1*i, current=1e5, center=np.array([1.0, 0, 0]), nfp=nfp)
    coils.append(coil)
    currents.append(Current(coil, 1e5))

# Define Biot-Savart solver
bs = BiotSavart(coils)
bs.set_points(surface.gamma())

# Define individual terms of the objective function
flux_objective = SquaredFlux(surface, bs)
coil_length = CoilLength(coils)
coil_to_coil_distance = MinimumDistanceCurveCurve(coils)
coil_to_surface_distance = MinimumDistanceCurveSurface(coils, surface)
curvature = Curvature(coils)
mean_squared_curvature = MeanSquaredCurvature(coils)

# Form the total objective function
objectives = [flux_objective, coil_length, coil_to_coil_distance, coil_to_surface_distance, curvature, mean_squared_curvature]
weights = [1.0, 0.1, 0.1, 0.1, 0.1, 0.1]  # Adjust weights as necessary
total_objective = LeastSquaresProblem(objectives, weights)

# Perform a Taylor test
least_squares_serial_solve(total_objective, method="lm", verbose=2)

# Run the optimization
result = least_squares_serial_solve(total_objective, method="lm", verbose=2)

# Use the result as the initial guess for a subsequent optimization with reduced penalty for the coil length
weights[1] = 0.01  # Reduce penalty for coil length
total_objective = LeastSquaresProblem(objectives, weights)
result_final = least_squares_serial_solve(total_objective, method="lm", verbose=2)

# Save the optimized coil shapes and currents
for i, coil in enumerate(coils):
    np.savetxt(f"coil_{i}_shape.txt", coil.gamma())
    np.savetxt(f"coil_{i}_current.txt", [currents[i].current])
```