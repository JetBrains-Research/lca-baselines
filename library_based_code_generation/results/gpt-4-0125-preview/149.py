```python
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, Coil, ScaledCurrent
from simsopt.objectives.fluxobjective import SquaredFlux, CoilLength
from simsopt.geo.curveobjectives import CurveLength
from simsopt.optimize.lbfgs import least_squares_serial_solve
from simsopt.field.magneticfieldclasses import InterpolatedField, MagneticField
from simsopt.field.tracing import VolumeAveragedBNormal
from simsopt.geo.curve import curves_to_vtk

# Define target equilibrium and surface
nfp = 5
nphi = 32
nzeta = 32
surface = SurfaceRZFourier(nfp, nphi, nzeta, "stellarator_sym")

# Load or define the target B field. Here we use a virtual casing principle to compute B_external.n
# For demonstration, we assume B_external is given or computed elsewhere.
# Replace this with the actual computation or loading of B_external.
B_external = None  # Placeholder for the external B field computation

# Define coils
num_coils = 6
coils = [Coil(Current(1e6), 100) for _ in range(num_coils)]
scaled_currents = [ScaledCurrent(c, 1.0) for c in coils]
bs = BiotSavart(coils)
bs.set_points(surface.gamma().reshape((-1, 3)))

# Define objective function
LENGTH_PENALTY = 1e-4
L0 = 1.0  # Target coil length, adjust as needed
objective = SquaredFlux(surface, bs) + LENGTH_PENALTY * sum(CoilLength(c) for c in coils)

# Perform Taylor test
from simsopt.util.taylortest import taylor_test
taylor_test(objective, dvecs=[*scaled_currents, surface], verbose=2)

# Run optimization
x0 = np.concatenate([c.currents for c in scaled_currents] + [surface.get_dofs()])
result = least_squares_serial_solve(objective, x0, method='L-BFGS-B', verbose=2)

# Save results
output_dir = "output"
surface.to_vtk(output_dir + "/optimized_surface.vtk")
curves_to_vtk(coils, output_dir + "/optimized_coils.vtk")
```