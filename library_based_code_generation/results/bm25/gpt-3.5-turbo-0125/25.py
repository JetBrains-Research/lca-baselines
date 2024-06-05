import simsopt

# Initialize boundary magnetic surface
boundary_surface = ...

# Create initial coils
initial_coils = ...

# Define objective function terms
magnetic_field_term = ...
coil_length_term = ...
coil_to_coil_distance_term = ...
coil_to_surface_distance_term = ...
curvature_term = ...
mean_squared_curvature_term = ...

# Form total objective function
objective_function = magnetic_field_term + coil_length_term + coil_to_coil_distance_term + coil_to_surface_distance_term + curvature_term + mean_squared_curvature_term

# Perform Taylor test
simsopt.test_curves_and_surface(boundary_surface, initial_coils)

# Run optimization
optimized_coils = simsopt.coil_optimization(objective_function, initial_coils)

# Use optimized coils as initial guess for subsequent optimization
reduced_penalty_coil_length = ...

# Save optimized coil shapes and currents
simsopt.save_optimized_coils(optimized_coils)