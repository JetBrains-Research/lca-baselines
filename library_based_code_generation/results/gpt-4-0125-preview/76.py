```python
import pybamm

# Create a new lithium-ion model
model = pybamm.lithium_ion.BasicDFN(name="my li-ion model")

# External circuit
model.submodels["external circuit"] = pybamm.external_circuit.ExternalCircuit(model.param)

# Current collector
model.submodels["current collector"] = pybamm.current_collector.Uniform(model.param)

# Thermal
model.submodels["thermal"] = pybamm.thermal.isothermal.Isothermal(model.param)

# Porosity
model.submodels["negative electrode porosity"] = pybamm.porosity.Constant(model.param, "Negative")
model.submodels["positive electrode porosity"] = pybamm.porosity.Constant(model.param, "Positive")

# Electrolyte diffusion
model.submodels["negative electrolyte diffusion"] = pybamm.electrolyte_diffusion.ConstantConcentration(model.param, "Negative")
model.submodels["positive electrolyte diffusion"] = pybamm.electrolyte_diffusion.ConstantConcentration(model.param, "Positive")

# Electrolyte conductivity
model.submodels["electrolyte conductivity"] = pybamm.electrolyte_conductivity.Constant(model.param)

# SEI
model.submodels["sei"] = pybamm.sei.NoSEI(model.param)

# SEI on cracks
model.submodels["sei on cracks"] = pybamm.sei.NoSEI(model.param)

# Lithium plating
model.submodels["lithium plating"] = pybamm.lithium_plating.NoPlating(model.param)

# Submodels for both negative and positive electrode domains
electrode_domains = ["Negative", "Positive"]
for domain in electrode_domains:
    prefix = domain.lower()
    model.submodels[f"{prefix} active material"] = pybamm.active_material.Constant(model.param, domain)
    model.submodels[f"{prefix} electrode potential"] = pybamm.electrode.ohm.LithiumMetalPotential(model.param, domain)
    model.submodels[f"{prefix} particle"] = pybamm.particle.ConstantConcentration(model.param, domain)
    model.submodels[f"{prefix} total particle concentration"] = pybamm.particle.ConstantConcentration(model.param, domain)
    model.submodels[f"{prefix} open circuit potential"] = pybamm.open_circuit_potential.NoOpenCircuitPotential(model.param, domain)
    model.submodels[f"{prefix} interface"] = pybamm.interface.InverseButlerVolmer(model.param, domain, "lithium-ion main")
    model.submodels[f"{prefix} interface utilisation"] = pybamm.interface_utilisation.Constant(model.param, domain)
    model.submodels[f"{prefix} interface current"] = pybamm.interface.CurrentForInverseButlerVolmer(model.param, domain, "lithium-ion main")
    model.submodels[f"{prefix} surface potential difference"] = pybamm.interface.SurfacePotentialDifference(model.param, domain, "lithium-ion main")
    model.submodels[f"{prefix} particle mechanics"] = pybamm.particle_mechanics.NoMechanics(model.param, domain)

# Build the model
model.build_model()

# Create geometry
geometry = model.default_geometry

# Process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# Set the mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# Discretise the model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# Solve the model
solver = pybamm.CasadiSolver()
t_eval = [0, 3600]  # 1 hour simulation
solution = solver.solve(model, t_eval)

# Plot the solution dynamically
plotter = pybamm.dynamic_plotter.DynamicPlot(solution)
plotter.dynamic_plot()
```