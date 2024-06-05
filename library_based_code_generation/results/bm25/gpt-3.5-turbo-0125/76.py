import pybamm

# Define custom lithium-ion model
model = pybamm.lithium_ion.BaseModel()

# Add submodels
model.submodels["external circuit"] = pybamm.external_circuit.CurrentControl()
model.submodels["current collector"] = pybamm.current_collector.Uniform()
model.submodels["thermal"] = pybamm.thermal.isothermal.Isothermal()
model.submodels["porosity"] = pybamm.porosity.Constant()
model.submodels["electrolyte diffusion"] = pybamm.electrolyte_diffusion.ConstantConcentration()
model.submodels["electrolyte conductivity"] = pybamm.electrolyte_conductivity.Constant()
model.submodels["SEI"] = pybamm.sei.NoSEI()
model.submodels["SEI on cracks"] = pybamm.sei.NoSEI()
model.submodels["lithium plating"] = pybamm.lithium_plating.NoPlating()

# Add submodels for negative and positive electrode domains
for domain in ["negative electrode", "positive electrode"]:
    model.submodels[domain + " active material"] = pybamm.active_material.Constant()
    model.submodels[domain + " electrode potential"] = pybamm.electrode.ohm.Full()
    model.submodels[domain + " particle"] = pybamm.particle.FickianSingleParticle()
    model.submodels[domain + " total particle concentration"] = pybamm.particle.total_concentration.Constant()
    model.submodels[domain + " open-circuit potential"] = pybamm.electrode.ohm.Full()
    model.submodels[domain + " interface"] = pybamm.interface.ButlerVolmer()
    model.submodels[domain + " interface utilisation"] = pybamm.interface.ReactionLimited()
    model.submodels[domain + " interface current"] = pybamm.interface.ButlerVolmer()
    model.submodels[domain + " surface potential difference"] = pybamm.interface.ReactionLimited()
    model.submodels[domain + " particle mechanics"] = pybamm.particle.mechanics.NoCompression()

# Build model
model.build_model()

# Create geometry
geometry = pybamm.Geometry()

# Process model and geometry
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
param.process_model(model)
param.process_geometry(geometry)

# Set mesh
mesh = pybamm.Mesh(geometry, model)

# Discretise model
disc = pybamm.Discretisation(mesh, model)
disc.process_model(model)

# Solve model
solver = pybamm.CasadiSolver()
t_eval = np.linspace(0, 3600, 100)
solution = solver.solve(model, t_eval)

# Plot solution dynamically
plot = pybamm.QuickPlot(solution)
plot.dynamic_plot()