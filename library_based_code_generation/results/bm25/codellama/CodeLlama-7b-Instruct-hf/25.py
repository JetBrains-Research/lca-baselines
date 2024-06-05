  import simsopt

# Initialize the boundary magnetic surface
surface = simsopt.test_curves_and_surface()

# Create initial coils
coils = [simsopt.CircularCoil(surface, 0.1, 0.5),
         simsopt.CircularCoil(surface, 0.2, 0.5),
         simsopt.CircularCoil(surface, 0.3, 0.5)]

# Define the individual terms of the objective function
B_term = simsopt.MagneticField(coils, surface)
L_term = simsopt.CoilLength(coils)
D_term = simsopt.CoilToCoilDistance(coils)
S_term = simsopt.CoilToSurfaceDistance(coils, surface)
K_term = simsopt.Curvature(coils)
MSK_term = simsopt.MeanSquaredCurvature(coils)

# Form the total objective function
objective = B_term + L_term + D_term + S_term + K_term + MSK_term

# Perform a Taylor test
test_result = simsopt.test_biotsavart_B_is_curlA(coils, surface)

# Run the optimization
optimization_result = simsopt.coil_optimization(coils, objective, surface,
                                                test_result=test_result)

# Use the result as the initial guess for a subsequent optimization
# with reduced penalty for the coil length
optimization_result_reduced = simsopt.coil_optimization(coils, objective, surface,
                                                        test_result=test_result,
                                                        penalty_coef=0.5)

# Save the optimized coil shapes and currents
simsopt.save_coils(optimization_result.coils, 'optimized_coils.json')
simsopt.save_currents(optimization_result.coils, 'optimized_currents.json')

# Plot the optimization results
simsopt.make_optimization_plots(optimization_result, 'optimization_results.png')
simsopt.make_optimization_plots(optimization_result_reduced, 'optimization_results_reduced.png')