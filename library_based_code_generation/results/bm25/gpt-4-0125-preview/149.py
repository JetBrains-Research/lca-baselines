```python
import numpy as np
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux, CoilLength
from simsopt.optimize.lbfgs import least_squares_serial_solve
from simsopt.field.virtualcasing import VirtualCasing
import os

# Load or define the target W7-X surface with an average beta of 4%
# Assuming the surface is defined in a file or can be generated. Here we create a dummy surface.
# In practice, you should replace this with the actual surface definition.
nfp, mpol, ntor = 5, 3, 2
surface = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor, sym=True)

# Define coils (this should be replaced with the actual coil configuration for W7-X)
# Here we create a dummy set of coils for demonstration.
coils = coils_via_symmetries(Current(100, np.linspace(1, 10, 10)), nfp, (True, False))

# Define the Biot-Savart solver
bs = BiotSavart(coils)
bs.set_points(surface.gamma())

# Define the target normal field using Virtual Casing
vc = VirtualCasing(surface, bs)
B_target = vc.Bnormal()

# Define the objective function
LENGTH_PENALTY = 1e-4
L0 = 1.0  # Target coil length, this should be set based on the problem specifics

def objective(x):
    for coil, xi in zip(coils, x.reshape(-1, 10)):
        coil.x = xi
    bs.clear_cached_properties()
    B_biot_savart = bs.B()
    B_normal = np.einsum('ijk,ik->ij', surface.normal(), B_biot_savart)
    flux_objective = SquaredFlux(surface, B_target, B_normal)
    length_objective = CoilLength(coils, L0)
    J = flux_objective.J() + LENGTH_PENALTY * length_objective.J()
    return J

# Perform Taylor test
from simsopt.util.taylortest import taylortest
x0 = np.array([coil.x for coil in coils]).flatten()
taylortest(objective, x0, h=1e-5)

# Run the optimization
result = least_squares_serial_solve(objective, x0, method='L-BFGS-B')

# Save the optimized coil parameters
if not os.path.exists("output"):
    os.makedirs("output")
np.save("output/optimized_coils.npy", result.x)

# Optionally, save other results or perform further analysis as needed.
```