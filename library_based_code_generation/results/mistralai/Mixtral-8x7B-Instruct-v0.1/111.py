 ```python
import numpy as np
from simsopt import Problem, Real, BoundaryValue, Objective, Constraint
from simsopt.geometry import Circle, Line, Spline
from simsopt.fields import BiotSavart
from scipy.optimize import minimize

# Define problem
problem = Problem()

# Define boundary magnetic surface
R0 = 3.0
a = 1.0
surface = Circle(0, R0 + a)

# Initialize boundary value
bval = BoundaryValue(surface)

# Define curves
ncurves = 20
curves = [Line(0, 2 * np.pi * i / ncurves) for i in range(ncurves)]
bval.add_curves(curves)

# Define multifilament grid
nmfilaments = 5
grid = np.linspace(0, 1, nmfilaments)
bval.add_grid(grid)

# Apply stellarator and rotation symmetries
for s in range(1, 4):
    problem.add_symmetry(bval, s)

# Define objective function
flux_weight = 1.0
curve_weight = 0.1
distance_weight = 0.01


def objective(x):
    bval.set_values(x)
    flux = BiotSavart(bval).integrate_flux()
    curve_length = BiotSavart(bval).integrate_curve_length()
    distance = BiotSavart(bval).integrate_distance()
    return flux_weight * flux**2 + curve_weight * curve_length + distance_weight * distance


problem.add_objective(Objective(objective))

# Define Taylor test
taylor_test = minimize(objective, bval.get_random_values(), method='SLSQP', bounds=bval.bounds)

# Run optimization
problem.set_initial_guess(taylor_test.x)
result = problem.minimize(method='L-BFGS-B')

# Save output in VTK format
bval.save_vtk('output.vtk')
```