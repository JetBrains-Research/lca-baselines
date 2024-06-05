import numpy as np
from simsopt import Simsopt

# Define the objective function
def objective_function(coil_params):
    # Compute Biot-Savart field
    B_biot_savart = compute_biot_savart(coil_params)
    
    # Compute external field using virtual casing calculation
    B_external = compute_external_field(coil_params)
    
    # Compute objective function value
    J = 0.5 * np.sum(np.abs(np.dot(B_biot_savart, n) - np.dot(B_external, n))**2) + LENGTH_PENALTY * np.sum(0.5 * (CurveLength - L0)**2)
    
    return J

# Initialize Simsopt object
simsopt = Simsopt(objective_function)

# Perform Taylor test
simsopt.taylor_test()

# Run optimization using L-BFGS-B method
result = simsopt.run_optimization(method='L-BFGS-B')

# Save results
result.save_results(output_dir='output')