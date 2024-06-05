 import simsopt as so
import numpy as np
from scipy.sparse.linalg import spsolve

# Define target equilibrium
target_beta = 0.04

# Define virtual casing parameters
virtual_casing_radius = 5.5  # [m]
virtual_casing_resolution = 50  # number of points on the casing

# Define length penalty
LENGTH_PENALTY = 1e-6
L0 = 1.0  # [m]

# Define the function to compute B_External . n
def b_external_dot_n(x, coil_geoms):
    b_ext = so.external_field(x, coil_geoms, virtual_casing_radius, virtual_casing_resolution)
    n = so.surface_normal(coil_geoms)
    return np.sum(b_ext * n, axis=1)

# Define the objective function
def obj_func(x, coil_geoms):
    b_ext_dot_n = b_external_dot_n(x, coil_geoms)
    b_dot_n = so.biot_savart_field(x, coil_geoms) * n
    integral_term = np.sum((b_dot_n - b_ext_dot_n)**2)
    length_term = LENGTH_PENALTY * np.sum(0.5 * (so.curve_length(coil_geoms) - L0)**2)
    return integral_term + length_term

# Define the gradient of the objective function
def obj_func_grad(x, coil_geoms):
    n = so.surface_normal(coil_geoms)
    b_ext_dot_n = b_external_dot_n(x, coil_geoms)
    b_dot_n = so.biot_savart_field(x, coil_geoms) * n
    dJdx = np.zeros_like(x)
    for i in range(coil_geoms.shape[0]):
        dJdx[i*6:(i+1)*6] = so.line_current_jacobian(x[i*6:(i+1)*6], coil_geoms[i])
    for i in range(coil_geoms.shape[0]):
        dJdx[i*6:(i+1)*6] += 2 * np.sum((b_dot_n[i] - b_ext_dot_n[i]) * so.biot_savart_field(x[i*6:(i+1)*6], coil_geoms[i]) * n[i], axis=0)
    length_jac = LENGTH_PENALTY * np.zeros_like(x)
    for i in range(coil_geoms.shape[0]):
        length_jac[i*6:(i+1)*6] = so.curve_length_jacobian(coil_geoms[i])
    dJdx += length_jac * (so.curve_length(coil_geoms) - L0)
    return dJdx

# Define the Hessian of the objective function
def obj_func_hess(x, coil_geoms):
    n = so.surface_normal(coil_geoms)
    b_ext_dot_n = b_external_dot_n(x, coil_geoms)
    b_dot_n = so.biot_savart_field(x, coil_geoms) * n
    H = np.zeros((x.shape[0], x.shape[0]))
    for i in range(coil_geoms.shape[0]):
        H_block = so.line_current_hessian(x[i*6:(i+1)*6], coil_geoms[i])
        H[i*6:(i+1)*6, i*6:(i+1)*6] = H_block
    for i in range(coil_geoms.shape[0]):
        for j in range(coil_geoms.shape[0]):
            H[i*6:(i+1)*6, j*6:(j+1)*6] += 2 * np.outer(b_dot_n[i], so.biot_savart_field(x[j*6:(j+1)*6], coil_geoms[j]) * n[j])
    length_hess = LENGTH_PENALTY * np.zeros((x.shape[0], x.shape[0]))
    for i in range(coil_geoms.shape[0]):
        length_hess[i*6:(i+1)*6, i*6:(i+1)*6] = so.curve_length_hessian(coil_geoms[i])
    H += length_hess
    return H

# Define the Taylor test function
def taylor_test(x, coil_geoms):
    x_trial = x + 1e-8 * np.random.randn(x.shape[0])
    f_trial = obj_func(x_trial, coil_geoms)
    df_trial = obj_func_grad(x_trial, coil_geoms)
    d2f_trial = obj_func_hess(x_trial, coil_geoms)
    d2f = obj_func_hess(x, coil_geoms)
    dfx = obj_func_grad(x, coil_geoms)
    return (f_trial - obj_func(x, coil_geoms) - df_trial.dot(x_trial - x)) / (0.5 * (x_trial - x).dot(d2f_trial @ (x_trial - x)) + 1e-16) - 1

# Define the optimization problem
problem = so.Problem()
problem.add_variables(coil_geoms, lower_bounds=-np.inf, upper_bounds=np.inf)
problem.add_objective(obj_func, grad_func=obj_func_grad, hess_func=obj_func_hess)
problem.add_constraint(so.total_current, equals=0)
problem.add_constraint(so.total_enclosed_flux, equals=0)
problem.add_constraint(so.minimum_distance, equals=0.1)
problem.add_constraint(taylor_test, equals=0)

# Run the optimization using the L-BFGS-B method
result = problem.minimize(method='l-bfgs-b', options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'maxcor': 10})

# Save the results in the "output" directory
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
np.savetxt(os.path.join(output_dir, "coil_geoms.txt"), result.x)
np.savetxt(os.path.join(output_dir, "objective_function.txt"), [result.fun])
np.savetxt(os.path.join(output_dir, "constraints.txt"), result.constr)