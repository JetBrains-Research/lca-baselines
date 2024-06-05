  ```
from pybamm import *

# Define the custom lithium-ion model
my_li_ion_model = Model(
    "my li-ion model",
    submodels=[
        ExternalCircuit(),
        CurrentCollector(),
        Thermal(),
        Porosity(),
        ElectrolyteDiffusion(),
        ElectrolyteConductivity(),
        SEI(),
        SEIOnCracks(),
        LithiumPlating(),
    ],
)

# Define the negative electrode domain submodels
my_li_ion_model.submodels.append(
    ActiveMaterial(domain="Negative electrode")
)
my_li_ion_model.submodels.append(
    ElectrodePotential(domain="Negative electrode")
)
my_li_ion_model.submodels.append(
    Particle(domain="Negative electrode")
)
my_li_ion_model.submodels.append(
    TotalParticleConcentration(domain="Negative electrode")
)
my_li_ion_model.submodels.append(
    OpenCircuitPotential(domain="Negative electrode")
)
my_li_ion_model.submodels.append(
    Interface(domain="Negative electrode")
)
my_li_ion_model.submodels.append(
    InterfaceUtilisation(domain="Negative electrode")
)
my_li_ion_model.submodels.append(
    InterfaceCurrent(domain="Negative electrode")
)
my_li_ion_model.submodels.append(
    SurfacePotentialDifference(domain="Negative electrode")
)
my_li_ion_model.submodels.append(
    ParticleMechanics(domain="Negative electrode")
)

# Define the positive electrode domain submodels
my_li_ion_model.submodels.append(
    ActiveMaterial(domain="Positive electrode")
)
my_li_ion_model.submodels.append(
    ElectrodePotential(domain="Positive electrode")
)
my_li_ion_model.submodels.append(
    Particle(domain="Positive electrode")
)
my_li_ion_model.submodels.append(
    TotalParticleConcentration(domain="Positive electrode")
)
my_li_ion_model.submodels.append(
    OpenCircuitPotential(domain="Positive electrode")
)
my_li_ion_model.submodels.append(
    Interface(domain="Positive electrode")
)
my_li_ion_model.submodels.append(
    InterfaceUtilisation(domain="Positive electrode")
)
my_li_ion_model.submodels.append(
    InterfaceCurrent(domain="Positive electrode")
)
my_li_ion_model.submodels.append(
    SurfacePotentialDifference(domain="Positive electrode")
)
my_li_ion_model.submodels.append(
    ParticleMechanics(domain="Positive electrode")
)

# Build the model
my_li_ion_model.build()

# Create the geometry
my_li_ion_model.create_geometry()

# Process the model and geometry
my_li_ion_model.process_model_and_geometry()

# Set the mesh
my_li_ion_model.set_mesh()

# Discretise the model
my_li_ion_model.discretise()

# Solve the model
my_li_ion_model.solve()

# Plot the solution dynamically
my_li_ion_model.plot()
```