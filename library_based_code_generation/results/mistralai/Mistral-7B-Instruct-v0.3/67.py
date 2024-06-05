 ```python
import pyaedt as aedt

# Set non-graphical mode
aedt.set_option('silent', True)

# Launch AEDT and 2D Extractor
aedt.launch()

# Define variables
frequency = 10.0e9
width = 1.0e-3
gap = 5.0e-6
length = 10.0e-3
thickness_coplanar = 1.0e-6
thickness_dielectric = 2.0e-6
thickness_coating = 1.0e-6
perm_dielectric = 2.2
perm_coating = 1.0
loss_tangent_coating = 0.0005

# Create primitives
signal = aedt.Line(start=(0, 0), end=(length, 0), width=width, layer='Signal')
coplanar_ground = aedt.Line(start=(0, 0), end=(length, 0), width=width+2*gap, layer='Ground')
reference_ground_plane = aedt.Rectangle(center=(0, 0), size=(length, length), layer='Ground')
dielectric = aedt.Rectangle(center=(length/2, 0), size=(width, length), thickness=thickness_dielectric,
                            material=aedt.Material(permittivity=perm_dielectric, loss_tangent=0.0))
coating = aedt.Rectangle(center=(length/2, 0), size=(width, length), thickness=thickness_coating,
                         material=aedt.Material(permittivity=perm_coating, loss_tangent=loss_tangent_coating))

# Create a signal
signal.assign_conductor()

# Create a reference ground
reference_ground = aedt.ReferenceGround(reference_ground_plane)

# Create the Huray model on the signal
signal.set_port(aedt.Port(name='Port1', location=(0, 0)))
signal.set_port_type('S11')
signal.set_port_model('Huray')

# Create the setup
setup = aedt.Setup(frequency=frequency, max_iterations=100, solver='MOM')

# Analyze it
analysis = aedt.Analysis(setup=setup, structures=[signal, coplanar_ground, reference_ground, dielectric, coating])
analysis.run()

# Plot solution data
aedt.plot_solution(analysis, solution='S11', x_data=frequency)

# Save the project
aedt.save_project('CPWG_Design.prj')

# Close AEDT
aedt.quit()
```