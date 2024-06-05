  import simsopt

# Define the problem parameters
surface = simsopt.Surface("path/to/surface.obj")
target_normal_field = 1000
target_equilibrium = "W7-X"
target_beta = 4
length_penalty = 100

# Define the objective function
def objective(coils):
    # Compute the target B_External
    B_External = simsopt.compute_B_External(coils, surface, target_normal_field)

    # Compute the B_BiotSavart
    B_BiotSavart = simsopt.compute_B_BiotSavart(coils, surface)

    # Compute the objective function
    J = 0.5 * simsopt.integrate(lambda s: (B_BiotSavart[s] - B_External[s])**2, surface)
    J += length_penalty * simsopt.integrate(lambda s: 0.5 * (surface.curve_length[s] - surface.curve_length[0])**2, surface)

    return J

# Define the constraints
def constraints(coils):
    # Compute the B_External
    B_External = simsopt.compute_B_External(coils, surface, target_normal_field)

    # Compute the B_BiotSavart
    B_BiotSavart = simsopt.compute_B_BiotSavart(coils, surface)

    # Compute the constraint violation
    violation = simsopt.integrate(lambda s: (B_BiotSavart[s] - B_External[s])**2, surface)

    # Compute the Taylor test
    taylor_test = simsopt.taylor_test(coils, surface, target_equilibrium, target_beta)

    return violation, taylor_test

# Define the optimization parameters
optimization_parameters = {
    "method": "L-BFGS-B",
    "max_iter": 100,
    "ftol": 1e-6,
    "gtol": 1e-6,
    "xtol": 1e-6,
    "disp": False
}

# Run the optimization
coils = simsopt.optimize(objective, constraints, optimization_parameters)

# Save the results
simsopt.save_coils("output/coils.obj", coils)
simsopt.save_surface("output/surface.obj", surface)
simsopt.save_results("output/results.csv", coils, surface, target_normal_field, target_equilibrium, target_beta)