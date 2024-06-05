import simsopt

# Define the target equilibrium parameters
target_equilibrium = {
    'configuration': 'W7-X',
    'average_beta': 0.04
}

# Define the objective function
def objective_function(B_BiotSavart_dot_n, B_External_dot_n, CurveLength, L0, LENGTH_PENALTY):
    return 0.5 * sum((B_BiotSavart_dot_n - B_External_dot_n)**2) + LENGTH_PENALTY * sum(0.5 * (CurveLength - L0)**2)

# Perform the coil optimization using L-BFGS-B method
result = simsopt.coil_optimization(target_equilibrium, objective_function, method='L-BFGS-B')

# Save the results in the "output" directory
result.save_results("output")