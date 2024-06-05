 ```python
import pyvista as pv
from pyvista.examples.data import load_sphere
from pyvista.utilities import download
import numpy as np

# Download and load planet textures
texture_urls = {
    'mercury': 'https://raw.githubusercontent.com/NASA-Imagery-API/Planetary-Pixel-Tiles/master/textures/mercury.jpg',
    'venus': 'https://raw.githubusercontent.com/NASA-Imagery-API/Planetary-Pixel-Tiles/master/textures/venus.jpg',
    'earth': 'https://raw.githubusercontent.com/NASA-Imagery-API/Planetary-Pixel-Tiles/master/textures/earth.jpg',
    'mars': 'https://raw.githubusercontent.com/NASA-Imagery-API/Planetary-Pixel-Tiles/master/textures/mars.jpg',
    'jupiter': 'https://raw.githubusercontent.com/NASA-Imagery-API/Planetary-Pixel-Tiles/master/textures/jupiter.jpg',
    'saturn': 'https://raw.githubusercontent.com/NASA-Imagery-API/Planetary-Pixel-Tiles/master/textures/saturn.jpg',
    'uranus': 'https://raw.githubusercontent.com/NASA-Imagery-API/Planetary-Pixel-Tiles/master/textures/uranus.jpg',
    'neptune': 'https://raw.githubusercontent.com/NASA-Imagery-API/Planetary-Pixel-Tiles/master/textures/neptune.jpg'
}
textures = {name: download(url) for name, url in texture_urls.items()}

# Load planet models and apply textures
planets = {}
for name, url in texture_urls.items():
    planet = load_sphere(name)
    planet.texture = pv.Texture(textures[name])
    planets[name] = planet

# Position planets in 3D space
solar_system = pv.MultiBlock()
solar_system.append(planets['sun'], deep_copy=False)

planet_positions = {
    'mercury': np.array([0.387, 0, 0]),
    'venus': np.array([0.723, 0, 0]),
    'earth': np.array([1, 0, 0]),
    'mars': np.array([1.524, 0, 0]),
    'jupiter': np.array([5.203, 0, 0]),
    'saturn': np.array([9.582, 0, 0]),
    'uranus': np.array([19.182, 0, 0]),
    'neptune': np.array([30.07, 0, 0])
}

for name, position in planet_positions.items():
    planet = planets[name]
    planet.rotate_z(np.radians(90))
    planet.translate(position)
    solar_system.append(planet, deep_copy=False)

# Create light source to simulate the sun
light_source = pv.Light()
light_source.intensity = 100
solar_system.light = light_source

# Create subplots for individual planets
for name, planet in planets.items():
    plotter = pv.Plotter(shape=(1, len(planets)), off_screen=True)
    plotter.subplot(0, planets.index(planet))
    plotter.add_mesh(planet, show_edges=False)
    plotter.show_grid()
    plotter.show_axes()
    plotter.link_view()
    plotter.screenshot(f'{name}.png')

# Create visualization of Venus with and without its atmosphere
venus_atmosphere = pv.Sphere(center=(0.723, 0, 0), radius=0.005, start_theta=0, end_theta=np.pi)
venus_atmosphere.alpha = 0.5
venus_with_atmosphere = pv.MultiBlock()
venus_with_atmosphere.append(planets['venus'], deep_copy=False)
venus_with_atmosphere.append(venus_atmosphere, deep_copy=False)

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(planets['venus'], show_edges=False)
plotter.add_mesh(venus_atmosphere, show_edges=False)
plotter.show_grid()
plotter.show_axes()
plotter.screenshot('venus_with_atmosphere.png')

plotter.clear()
plotter.add_mesh(planets['venus'], show_edges=False)
plotter.show_grid()
plotter.show_axes()
plotter.screenshot('venus_without_atmosphere.png')

# Display the solar system
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(solar_system, show_edges=False)
plotter.show_grid()
plotter.show_axes()
plotter.link_view()
plotter.screenshot('solar_system.png')
```