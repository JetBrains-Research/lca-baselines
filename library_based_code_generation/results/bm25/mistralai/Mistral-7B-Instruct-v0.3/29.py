 Here is a basic example of how you might create a 3D visualization of the solar system using PyVista. Note that this code assumes you have the necessary 3D models for the planets and their textures downloaded and saved in the correct format (STL for 3D models and PNG for textures).

```python
import pyvista as pv
import numpy as np
import time

# Load the sun
sun = pv.read('sun.stl')
sun.texture = pv.Texture('sun_texture.png')

# Load the planets
mercury = pv.read('mercury.stl')
mercury.texture = pv.Texture('mercury_texture.png')
venus = pv.read('venus.stl')
venus_atmosphere = pv.read('venus_atmosphere.stl')
venus.texture = pv.Texture('venus_texture.png')
venus_atmosphere.texture = pv.Texture('venus_atmosphere_texture.png')
earth = pv.read('earth.stl')
earth.texture = pv.Texture('earth_texture.png')
mars = pv.read('mars.stl')
mars.texture = pv.Texture('mars_texture.png')
jupiter = pv.read('jupiter.stl')
jupiter.texture = pv.Texture('jupiter_texture.png')
saturn = pv.read('saturn.stl')
saturn.texture = pv.Texture('saturn_texture.png')
uranus = pv.read('uranus.stl')
uranus.texture = pv.Texture('uranus_texture.png')
neptune = pv.read('neptune.stl')
neptune.texture = pv.Texture('neptune_texture.png')

# Create a light source to simulate the sun
light = pv.PointLightSource(position=(0, 0, 0), intensity=1000)

# Create a plotter
plotter = pv.Plotter(size=(800, 800))

# Add the sun and light source to the plotter
plotter.add_light(light)
plotter.add_mesh(sun, background=True)

# Position the planets in 3D space
mercury.translate((30, 0, 0))
venus.translate((60, 0, 0))
venus_atmosphere.translate((60, 0, 0))
earth.translate((93, 0, 0))
mars.translate((140, 0, 0))
jupiter.translate((220, 0, 0))
saturn.translate((280, 0, 0))
uranus.translate((340, 0, 0))
neptune.translate((390, 0, 0))

# Add the planets to the plotter
for planet in [mercury, venus, venus_atmosphere, earth, mars, jupiter, saturn, uranus, neptune]:
    plotter.add_mesh(planet)

# Create subplots for individual planets
for i, planet in enumerate([mercury, venus, venus_atmosphere, earth, mars, jupiter, saturn, uranus, neptune]):
    plotter.subplot(1, len(planet), i+1)
    plotter.add_mesh(planet)

# Visualization of Venus with and without its atmosphere
plotter.subplot(2, 1, 1)
plotter.add_mesh(venus)
plotter.subplot(2, 1, 2)
plotter.add_mesh(venus_atmosphere)

# Show the plot
plotter.show()
```

This code assumes that the 3D models and textures for the planets are in the same directory as the script. You may need to adjust the positions of the planets to better represent the solar system. Also, you may need to adjust the light intensity and position to better simulate the sun.