import pyvista as pv

# Load models of the planets
earth = pv.read('earth.obj')
mars = pv.read('mars.obj')
venus = pv.read('venus.obj')

# Apply textures to the planets
earth.texture_map_to_plane(inplace=True)
mars.texture_map_to_plane(inplace=True)
venus.texture_map_to_plane(inplace=True)

# Position the planets in 3D space
earth.pos = [0, 0, 0]
mars.pos = [1.5, 0, 0]
venus.pos = [3, 0, 0]

# Create a light source to simulate the sun
light = pv.Light(position=[0, 0, 0])

# Create a plotter and add the models
plotter = pv.Plotter()
plotter.add_mesh(earth, texture=True)
plotter.add_mesh(mars, texture=True)
plotter.add_mesh(venus, texture=True)
plotter.add_light(light)

# Display the plotter
plotter.show()

# Create subplots for individual planets
plotter.subplot(0, 0)
plotter.add_mesh(earth, texture=True)
plotter.subplot(0, 1)
plotter.add_mesh(mars, texture=True)
plotter.subplot(1, 0)
plotter.add_mesh(venus, texture=True)

# Create visualization of Venus with and without atmosphere
plotter.subplot(1, 1)
plotter.add_mesh(venus, texture=True)
plotter.add_mesh(venus, texture=True, scalars='atmosphere')
plotter.show()