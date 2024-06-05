 Here is a simplified Python code using the simsopt library to solve the given problem. Please note that you might need to adjust the code according to your specific setup and import necessary modules.

```python
import numpy as np
from simsopt import *
from simsopt.biot_savart import BiotSavart
from simsopt.virtual_casing import VirtualCasing

# Define the problem dimensions
n_coils = 10
n_variables = n_coils * 3

# Define the target normal field B_External
B_External = np.zeros((3, n_coils))
# Fill in the values for B_External based on the W7-X configuration

# Define the virtual casing calculation
virtual_casing = VirtualCasing(surface_mesh, boundary_conditions)

# Define the Biot-Savart operator
biot_savart = BiotSavart(virtual_casing.A)

# Define the objective function and its gradient
def obj_func(X):
    A = biot_savart.A(X)
    J = 0.5 * np.sum((biot_savart.B(A) * np.outer(virtual_casing.n, 1) - B_External * np.outer(virtual_casing.n, 1))**2 * virtual_casing.ds + n_coils * np.sqrt(np.sum((A - virtual_casing.A)**2, axis=1)))
    return J

def grad_obj_func(X):
    A = biot_savart.A(X)
    grad_J = np.zeros((n_variables,))
    grad_J[0] = 0.5 * np.sum((biot_savart.B(A) * np.outer(virtual_casing.n, 1) - B_External * np.outer(virtual_casing.n, 1))**2 * virtual_casing.ds * biot_savart.A_and_dA_vjp(A, X)[:, 0] + n_coils * np.sqrt(np.sum((A - virtual_casing.A)**2, axis=1)) * biot_savart._dA_by_dX_impl(A, X)[0])
    for i in range(1, n_variables):
        grad_J[i] = 0.5 * np.sum((biot_savart.B(A) * np.outer(virtual_casing.n, 1) - B_External * np.outer(virtual_casing.n, 1))**2 * virtual_casing.ds * biot_savart.A_and_dA_vjp(A, X)[i] + n_coils * np.sqrt(np.sum((A - virtual_casing.A)**2, axis=1)) * biot_savart._dA_by_dX_impl(A, X)[i])
    return grad_J

# Define the Taylor test
def taylor_test(X, Y):
    A_X = biot_savart.A(X)
    A_Y = biot_savart.A(Y)
    return np.linalg.norm(A_X - A_Y)

# Define the problem constraints
def is_free(X):
    return np.all(X >= -1e6)

def is_fixed(X):
    return np.all(X <= 1e6)

# Define the optimization problem
problem = Problem(n_variables, obj_func, grad_obj_func, is_free, is_fixed, taylor_test)

# Run the optimization using the L-BFGS-B method
options = OptimizationOptions(max_iter=1000, ftol=1e-6, xtol=1e-6)
result = coil_optimization(problem, options, method='L-BFGS-B')

# Save the results
np.save("output/coil_currents", result.x)
```

This code assumes that you have defined the surface mesh, boundary conditions, and other necessary variables for the problem. The objective function, gradient, Taylor test, and constraints are defined according to the given instructions. The optimization problem is then solved using the L-BFGS-B method, and the results are saved in the "output" directory.