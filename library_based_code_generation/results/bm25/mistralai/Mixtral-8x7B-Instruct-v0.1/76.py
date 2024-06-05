 from pybamm import Model, SolveModel, set_up_and_parameterise_model_for_experiment, \
_check_and_prepare_model_inplace, TimeBuildModelParticle, test_model_solver_python_and_jax, \
process_parameters_and_discretise, check_and_set_domains, process_and_check, build_model, \
_build_model, TimeBuildModelSEI, test_shape_and_size_for_testing, test_process_parameters_and_discretise, \
variables_and_events, process_model, process_rhs_and_algebraic, TimeBuildModelLithiumPlating

custom_model = Model("my li-ion model")

# Submodels for external circuit
custom_model.submodels.external_circuit = {}

# Submodels for current collector
custom_model.submodels.current_collector = {}

# Submodels for thermal
custom_model.submodels.thermal = {}

# Submodels for porosity
custom_model.submodels.porosity = {}

# Submodels for electrolyte diffusion
custom_model.submodels.electrolyte_diffusion = {}

# Submodels for electrolyte conductivity
custom_model.submodels.electrolyte_conductivity = {}

# Submodels for SEI
custom_model.submodels.SEI = {}

# Submodels for SEI on cracks
custom_model.submodels.SEI_on_cracks = {}

# Submodels for lithium plating
custom_model.submodels.lithium_plating = {}

# Submodels for negative electrode domains
custom_model.submodels.negative_electrode = {
    "active_material": {},
    "electrode_potential": {},
    "particle": {},
    "total_particle_concentration": {},
    "open_circuit_potential": {},
    "interface": {},
    "interface_utilisation": {},
    "interface_current": {},
    "surface_potential_difference": {},
    "particle_mechanics": {},
}

# Submodels for positive electrode domains
custom_model.submodels.positive_electrode = {
    "active_material": {},
    "electrode_potential": {},
    "particle": {},
    "total_particle_concentration": {},
    "open_circuit_potential": {},
    "interface": {},
    "interface_utilisation": {},
    "interface_current": {},
    "surface_potential_difference": {},
    "particle_mechanics": {},
}

build_model(custom_model)

# Create the geometry
geometry = custom_model.get_geometry()

# Process the model and geometry
process_and_check(custom_model, geometry)

# Set the mesh
custom_model.set_mesh("custom")

# Discretise the model
process_parameters_and_discretise(custom_model, geometry)

# Solve the model
solver_results = solve_model(custom_model)

# Plot the solution dynamically
custom_model.plot_solution(solver_results)