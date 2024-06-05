import pybamm

# Create custom lithium-ion model
model = pybamm.lithium_ion.DFN()
model.name = "my li-ion model"

# Add submodels
model.submodels["external circuit"] = pybamm.external_circuit.CurrentControl()
model.submodels["negative electrode"] = pybamm.electrode.ohm.leading_order.CurrentCollector()
model.submodels["positive electrode"] = pybamm.electrode.ohm.leading_order.CurrentCollector()
model.submodels["thermal"] = pybamm.thermal.isothermal.Isothermal()
model.submodels["porosity"] = pybamm.porosity.Constant()
model.submodels["electrolyte diffusion"] = pybamm.electrolyte.stefan_maxwell.isotropic.Fast()
model.submodels["electrolyte conductivity"] = pybamm.electrolyte.conductivity.stefan_maxwell.isotropic.Fast()
model.submodels["sei"] = pybamm.sei.NoSEI()
model.submodels["sei film resistance"] = pybamm.sei.NoSEI()
model.submodels["lithium plating"] = pybamm.lithium_plating.NoPlating()

# Define geometry
geometry = model.default_geometry

# Process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# Set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# Discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# Solve model
solver = pybamm.CasadiSolver()
solution = solver.solve(model)

# Plot solution dynamically
plot = pybamm.QuickPlot(solution)
plot.dynamic_plot()