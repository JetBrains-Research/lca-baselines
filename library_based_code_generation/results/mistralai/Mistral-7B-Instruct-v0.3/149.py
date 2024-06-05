 Here is a simplified Python code using the `scipy.optimize` and `numpy` libraries to solve the stage-II coil optimization problem. Please note that this is a simplified version and might need adjustments based on your specific requirements.

```python
import numpy as np
from scipy.optimize import minimize
from simsopt import Coil, Surface, BiotSavart, VirtualCasing, L2Norm

# Define the surface and virtual casing
surface = Surface(np.load('surface_data.npy'))
virtual_casing = VirtualCasing(surface, 0.1)

# Define the coil and Biot-Savart law
coil = Coil(n_turns=100, wire_diameter=0.001)
bs = BiotSavart(coil, virtual_casing)

# Define the target external field and length penalty
target_B_external = np.ones(surface.n_points) * 0.04
L0 = 1.0  # desired curve length
length_penalty = 0.1  # weight for length penalty

# Define the objective function and its gradient
def obj_func(x):
    coil.positions = x
    B_BiotSavart = bs(coil.positions)
    B_external_dot_n = np.dot(target_B_external, surface.normals)
    B_BiotSavart_dot_n = np.dot(B_BiotSavart, surface.normals)
    error = L2Norm()(B_BiotSavart_dot_n - B_external_dot_n)
    length_error = np.sum(0.5 * (coil.curve_length(coil.positions) - L0)**2)
    return error + length_penalty * length_error

def grad_obj_func(x):
    coil.positions = x
    B_BiotSavart = bs(coil.positions)
    B_BiotSavart_dot_n = np.dot(B_BiotSavart, surface.normals)
    error = L2Norm()(B_BiotSavart_dot_n - B_external_dot_n)
    length_error = np.sum(coil.curve_length(coil.positions) - L0)
    grad_error = L2Norm().grad(B_BiotSavart_dot_n - B_external_dot_n)
    grad_length_error = 2 * length_penalty * coil.curve_length(coil.positions) - 2 * length_penalty * L0
    return np.concatenate((grad_error, grad_length_error))

# Define the Taylor test
def taylor_test(fun, x, v, eps=1e-8):
    return np.linalg.norm(fun(x + eps * v) - fun(x) - np.dot(eps, fun.grad(x))) < eps * np.linalg.norm(v)

# Run the optimization with L-BFGS-B method
bounds = ((-1, 1) for _ in range(coil.n_coils))
result = minimize(obj_func, coil.positions.flatten(), jac=grad_obj_func, method='L-BFGS-B', bounds=bounds)

# Save the results
np.save('coil_positions', result.x.reshape(coil.n_coils, 3))
```

This code assumes that you have a pre-defined surface data in a numpy array and that the surface normals are correctly computed. The coil positions are saved in a numpy array named 'coil_positions' in the 'output' directory after running the optimization.