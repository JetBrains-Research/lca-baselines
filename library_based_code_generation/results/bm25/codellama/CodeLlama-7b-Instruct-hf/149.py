  Here is the Python code that solves the stage-II coil optimization problem using the simsopt library:
```
import simsopt
import numpy as np

# Define the target normal field on the surface
target_normal_field = np.array([0, 0, 1])

# Define the virtual casing calculation to compute the target B_External·n
def virtual_casing_calculation(coils):
    # Compute the B_BiotSavart vector field
    B_BiotSavart = simsopt.A(coils)
    
    # Compute the B_External vector field
    B_External = np.cross(target_normal_field, B_BiotSavart)
    
    # Compute the B_External·n vector field
    B_External_n = np.dot(B_External, target_normal_field)
    
    return B_External_n

# Define the objective function
def objective_function(coils):
    # Compute the B_BiotSavart vector field
    B_BiotSavart = simsopt.A(coils)
    
    # Compute the B_External vector field
    B_External = np.cross(target_normal_field, B_BiotSavart)
    
    # Compute the B_External·n vector field
    B_External_n = np.dot(B_External, target_normal_field)
    
    # Compute the integral of the difference between B_BiotSavart and B_External·n
    integral = np.sum(np.abs(B_BiotSavart - B_External_n)**2)
    
    # Add the length penalty term
    integral += LENGTH_PENALTY * np.sum(np.abs(simsopt.coils_to_makegrid(coils) - L0)**2)
    
    return integral

# Define the Taylor test
def taylor_test(coils):
    # Compute the B_BiotSavart vector field
    B_BiotSavart = simsopt.A(coils)
    
    # Compute the B_External vector field
    B_External = np.cross(target_normal_field, B_BiotSavart)
    
    # Compute the B_External·n vector field
    B_External_n = np.dot(B_External, target_normal_field)
    
    # Compute the difference between B_BiotSavart and B_External·n
    difference = np.abs(B_BiotSavart - B_External_n)
    
    # Compute the Taylor test
    taylor_test = np.sum(difference**2)
    
    return taylor_test

# Define the optimization problem
problem = simsopt.coil_optimization(
    objective_function=objective_function,
    taylor_test=taylor_test,
    virtual_casing_calculation=virtual_casing_calculation,
    coil_optimization_method='L-BFGS-B',
    output_directory='output'
)

# Run the optimization
results = problem.solve()

# Save the results
results.save('output/results.csv')
```
This code defines the target normal field on the surface, the virtual casing calculation to compute the target B_External·n, the objective function, the Taylor test, and the optimization problem. It then runs the optimization using the L-BFGS-B method and saves the results to a CSV file in the "output" directory.