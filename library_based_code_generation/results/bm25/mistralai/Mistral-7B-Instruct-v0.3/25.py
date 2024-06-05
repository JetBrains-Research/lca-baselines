 Here is a basic structure of the Python code using the simsopt library to solve the coil optimization problem. Please note that this is a simplified version and you may need to adjust it according to your specific requirements.

```python
from simsopt import *
from simsopt.geometry import *
from simsopt.coil_optimization import *

# Define the target normal field on the surface
target_normal_field = ...

# Define the surface
surface = test_curves_and_surface(...)

# Define the initial coils
initial_coils = [CircularCoil(center=Point3(0, 0, 0), radius=0.1) for _ in range(num_coils)]

# Define the individual terms of the objective function
def magnetic_field(coils):
    return MagneticField(coils, surface).norm()

def coil_length(coils):
    return sum([coil.length for coil in coils])

def coil_to_coil_distance(coils):
    distances = []
    for i in range(len(coils)):
        for j in range(i+1, len(coils)):
            distances.append((coils[i].center - coils[j].center).norm())
    return sum(distances)

def coil_to_surface_distance(coils):
    distances = []
    for coil in coils:
        distances.append(min([(coil.center - point).norm() for point in surface.points]))
    return sum(distances)

def curvature(coils):
    curvatures = []
    for point in surface.points:
        b = test_biotsavart_B_is_curlA(coils, point)
        curvatures.append(b[2])
    return sum(curvatures)

def mean_squared_curvature(coils):
    curvatures = []
    for point in surface.points:
        b = test_biotsavart_B_is_curlA(coils, point)
        curvatures.append(b[2]**2)
    return sum(curvatures)

# Form the total objective function
def objective(coils):
    return 10 * magnetic_field(coils) + 1 * coil_length(coils) + 1 * coil_to_coil_distance(coils) + 1 * coil_to_surface_distance(coils) + 10 * curvature(coils) + 10 * mean_squared_curvature(coils) + 10 * (target_normal_field - test_biotsavart_B_is_curlA(coils, surface.center))**2

# Perform the optimization
optimization_setup = _optimization_setup(initial_coils, A_and_dA_vjp, A_vjp, _A_impl, objective)
optimization_result = optimize(optimization_setup, num_iterations=100, tolerance=1e-6)

# Use the result as the initial guess for a subsequent optimization with reduced penalty for the coil length
reduced_objective = lambda coils: objective(coils) + 0.1 * coil_length(coils)
reduced_optimization_result = optimize(optimize(optimization_setup, num_iterations=100, tolerance=1e-6).optimizer.state, reduced_objective, num_iterations=100, tolerance=1e-6)

# Save the optimized coil shapes and currents
save_coil_shapes_and_currents(initial_coils, reduced_optimization_result.optimizer.state.x, "optimized_coils.npz")
```

This code initializes the boundary magnetic surface, creates initial coils, defines the individual terms of the objective function, and forms the total objective function. It then performs a Taylor test and runs the optimization. After the optimization, it uses the result as the initial guess for a subsequent optimization with reduced penalty for the coil length. Finally, it saves the optimized coil shapes and currents.

Please note that you need to replace the `...` with the actual definitions of the target normal field, the surface, and the number of coils. Also, you may need to adjust the optimization settings such as the number of iterations and the tolerance according to your specific problem.

The `save_coil_shapes_and_currents` function is not a part of the simsopt library and should be defined according to your specific file format requirements.