```python
import pyvista as pv
from pyvista import examples

# Create a plotter
plotter = pv.Plotter(shape=(4, 2))

# Dictionary to hold planet information
planets_info = {
    "Mercury": {"position": (0, 0, 0), "texture": examples.planets.download_mercury()},
    "Venus": {"position": (1, 0, 0), "texture": examples.planets.download_venus()},
    "Earth": {"position": (2, 0, 0), "texture": examples.planets.download_earth()},
    "Mars": {"position": (3, 0, 0), "texture": examples.planets.download_mars()},
    "Jupiter": {"position": (4, 0, 0), "texture": examples.planets.download_jupiter()},
    "Saturn": {"position": (5, 0, 0), "texture": examples.planets.download_saturn()},
    "Uranus": {"position": (6, 0, 0), "texture": examples.planets.download_uranus()},
    "Neptune": {"position": (7, 0, 0), "texture": examples.planets.download_neptune()},
}

# Create a light source to simulate the Sun
light = pv.Light(position=(0, 0, 1), light_type='directional', intensity=1)
plotter.add_light(light)

# Main solar system visualization
plotter.subplot(0, 0)
for name, info in planets_info.items():
    sphere = pv.Sphere(radius=0.1, center=info['position'])
    plotter.add_mesh(sphere, texture=info['texture'], name=name)
plotter.view_isometric()

# Subplots for individual planets
for i, (name, info) in enumerate(planets_info.items(), start=1):
    plotter.subplot(i // 4, i % 4)
    sphere = pv.Sphere(radius=0.1)
    plotter.add_mesh(sphere, texture=info['texture'], name=name)
    plotter.view_isometric()

# Venus with and without atmosphere
venus_atmosphere_texture = examples.planets.download_venus_atmosphere()

# Venus without atmosphere
plotter.subplot(3, 0)
sphere = pv.Sphere(radius=0.1)
plotter.add_mesh(sphere, texture=planets_info['Venus']['texture'], name="Venus without Atmosphere")
plotter.view_isometric()

# Venus with atmosphere
plotter.subplot(3, 1)
sphere = pv.Sphere(radius=0.1)
plotter.add_mesh(sphere, texture=venus_atmosphere_texture, name="Venus with Atmosphere")
plotter.view_isometric()

# Display the plotter
plotter.show()
```