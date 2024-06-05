import simsopt

# Initialize boundary magnetic surface
boundary_surface = simsopt.BoundarySurface()

# Create initial coils
initial_coils = simsopt.Coils()

# Define objective function terms
magnetic_field_term = simsopt.MagneticFieldTerm()
coil_length_term = simsopt.CoilLengthTerm()
coil_to_coil_distance_term = simsopt.CoilToCoilDistanceTerm()
coil_to_surface_distance_term = simsopt.CoilToSurfaceDistanceTerm()
curvature_term = simsopt.CurvatureTerm()
mean_squared_curvature_term = simsopt.MeanSquaredCurvatureTerm()

# Form total objective function
objective_function = simsopt.ObjectiveFunction()
objective_function.add_term(magnetic_field_term)
objective_function.add_term(coil_length_term)
objective_function.add_term(coil_to_coil_distance_term)
objective_function.add_term(coil_to_surface_distance_term)
objective_function.add_term(curvature_term)
objective_function.add_term(mean_squared_curvature_term)

# Perform Taylor test
taylor_test = simsopt.TaylorTest(objective_function)

# Run optimization
optimization = simsopt.Optimization(objective_function)

# Use optimized result as initial guess for subsequent optimization
reduced_penalty_coil_length_term = simsopt.CoilLengthTerm(penalty=0.5)
objective_function_reduced_penalty = simsopt.ObjectiveFunction()
objective_function_reduced_penalty.add_term(magnetic_field_term)
objective_function_reduced_penalty.add_term(reduced_penalty_coil_length_term)
objective_function_reduced_penalty.add_term(coil_to_coil_distance_term)
objective_function_reduced_penalty.add_term(coil_to_surface_distance_term)
objective_function_reduced_penalty.add_term(curvature_term)
objective_function_reduced_penalty.add_term(mean_squared_curvature_term)

optimization_reduced_penalty = simsopt.Optimization(objective_function_reduced_penalty)

# Save optimized coil shapes and currents
optimized_coils = optimization.get_optimized_coils()
optimized_coils.save("optimized_coils.txt")