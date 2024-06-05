  import simsopt

# Initialize the boundary magnetic surface
surface = simsopt.Surface(name='Target Surface')

# Create initial coils
coils = [simsopt.Coil(name='Coil {}'.format(i)) for i in range(10)]

# Define objective function terms
magnetic_field_term = simsopt.Term(name='Magnetic Field',
                                   expression=simsopt.Expression('-1/2*B^2', B=simsopt.Field('Magnetic', surface=surface)))

length_term = simsopt.Term(name='Coil Length',
                           expression=simsopt.Expression('-1/2*L^2', L=simsopt.Variable('Coil Length')))

distance_term = simsopt.Term(name='Coil-to-Coil Distance',
                             expression=simsopt.Expression('-1/2*D^2', D=simsopt.Variable('Coil-to-Coil Distance')))

surface_distance_term = simsopt.Term(name='Coil-to-Surface Distance',
                                     expression=simsopt.Expression('-1/2*S^2', S=simsopt.Variable('Coil-to-Surface Distance')))

curvature_term = simsopt.Term(name='Curvature',
                              expression=simsopt.Expression('-1/2*K^2', K=simsopt.Variable('Curvature')))

mean_squared_curvature_term = simsopt.Term(name='Mean Squared Curvature',
                                           expression=simsopt.Expression('-1/2*M^2', M=simsopt.Variable('Mean Squared Curvature')))

# Define total objective function
objective_function = simsopt.ObjectiveFunction(name='Total Objective Function',
                                              terms=[magnetic_field_term, length_term, distance_term, surface_distance_term, curvature_term, mean_squared_curvature_term])

# Perform Taylor test
taylor_test = simsopt.TaylorTest(name='Taylor Test',
                                objective_function=objective_function,
                                coils=coils,
                                surface=surface)

# Run optimization
optimization = simsopt.Optimization(name='Optimization',
                                   objective_function=objective_function,
                                   coils=coils,
                                   surface=surface,
                                   taylor_test=taylor_test)

# Reduce penalty for coil length in subsequent optimization
optimization.penalty_reduction_factor = 0.5

# Save optimized coil shapes and currents
optimization.save_coil_shapes_and_currents(filename='optimized_coils.csv')

# Run optimization with reduced penalty for coil length
optimization.run()

# Save optimized coil shapes and currents
optimization.save_coil_shapes_and_currents(filename='optimized_coils_with_reduced_penalty.csv')