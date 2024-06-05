```python
import pyvista as pv
from pyvista import examples

# Create a plotter
plotter = pv.Plotter(shape=(3, 3))

# Solar system planets with approximate average distances from the sun in AU for positioning (not to scale)
planets_info = {
    "Mercury": {"distance": 0.39, "texture": "mercury.jpg"},
    "Venus": {"distance": 0.72, "texture": "venus.jpg"},
    "Earth": {"distance": 1, "texture": "earth.jpg"},
    "Mars": {"distance": 1.52, "texture": "mars.jpg"},
    "Jupiter": {"distance": 5.2, "texture": "jupiter.jpg"},
    "Saturn": {"distance": 9.58, "texture": "saturn.jpg"},
    "Uranus": {"distance": 19.22, "texture": "uranus.jpg"},
    "Neptune": {"distance": 30.05, "texture": "neptune.jpg"},
}

# Load models and textures, then position them
for i, (planet, info) in enumerate(planets_info.items()):
    # Create a sphere for the planet
    sphere = pv.Sphere(radius=0.1, center=(info["distance"], 0, 0))
    
    # Load texture
    texture = examples.download_texture(info["texture"])
    
    # Add to subplot
    plotter.subplot(i // 3, i % 3)
    plotter.add_text(planet, font_size=10)
    plotter.add_mesh(sphere, texture=texture)

# Create a light source to simulate the sun
light = pv.Light(position=(0, 0, 0), light_type='point')
plotter.add_light(light)

# Display the solar system
plotter.show()

# Venus with and without atmosphere
venus_texture = examples.download_texture("venus.jpg")
venus_atmosphere_texture = examples.download_texture("venus_atmosphere.jpg")

# Venus without atmosphere
plotter = pv.Plotter()
sphere = pv.Sphere(radius=0.1, center=(0, 0, 0))
plotter.add_mesh(sphere, texture=venus_texture)
plotter.show()

# Venus with atmosphere
plotter = pv.Plotter()
sphere = pv.Sphere(radius=0.105, center=(0, 0, 0))  # Slightly larger sphere for atmosphere
plotter.add_mesh(sphere, texture=venus_atmosphere_texture)
plotter.show()
```