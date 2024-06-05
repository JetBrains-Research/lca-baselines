 Here is a simplified Python code using the `scipy.optimize` and `numpy` libraries to solve the coil optimization problem. Please note that this is a basic example and might need adjustments to fit your specific problem.

```python
import numpy as np
from scipy.optimize import minimize
from simsopt import Surface, Coil, Objective, Optimizer

# Initialize boundary magnetic surface
surface = Surface(np.array([...]), name='surface')

# Create initial coils
coils = [Coil(np.array([[0, 0], [1, 0]]), 1, name='coil1')
         for _ in range(num_coils)]

# Define individual terms of the objective function
def magnetic_field(coils, surface):
    return np.linalg.norm(surface.H(coils))

def coil_length(coil):
    return np.linalg.norm(coil.winding)

def coil_to_coil_distance(coils):
    return np.sum([np.linalg.norm(coil1.center - coil2.center) for coil1, coil2 in zip(coils, coils[1:])])

def coil_to_surface_distance(coils, surface):
    return np.sum([np.linalg.norm(coil.center - surface.points) for coil in coils])

def curvature(coils, surface):
    return np.mean(np.abs(surface.K(coils)))

def mean_squared_curvature(coils, surface):
    return np.mean(np.square(surface.K(coils)))

# Form the total objective function
def objective_function(coils, surface, target_field, penalty_coil_length, penalty_curvature):
    return magnetic_field(coils, surface) + penalty_coil_length * coil_length(coils[0]) + \
           penalty_curvature * (curvature(coils, surface) + mean_squared_curvature(coils, surface))

# Perform a Taylor test and run the optimization
taylor = Optimizer.Taylor(coils, surface, objective_function, options={'maxiter': 100})
result = taylor.solve(method='SLSQP', args=(surface, target_field, 10, 100))

# Use the result as the initial guess for a subsequent optimization with reduced penalty for the coil length
reduced_penalty = objective_function.copy()
reduced_penalty.args = (surface, target_field, 1, 100)
reduced_optimizer = Optimizer.Taylor(coils, surface, reduced_penalty, options={'maxiter': 100})
reduced_result = reduced_optimizer.solve(method='SLSQP', x0=result.x)

# Save the optimized coil shapes and currents
np.save('optimized_coil_shapes.npy', coils[0].winding)
np.save('optimized_coil_currents.npy', coils[0].current)
```

This code assumes that you have already defined the `Surface` class with methods `H`, `K`, and `points`. Also, replace the `np.array([...])` with your actual surface points, and adjust the `num_coils`, `target_field`, `penalty_coil_length`, and `penalty_curvature` variables according to your problem.