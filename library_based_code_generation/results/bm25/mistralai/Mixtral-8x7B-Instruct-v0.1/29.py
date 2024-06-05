 import pyvista as pv
from pyvista.examples import examples_dir
import numpy as np

# Load models of the planets
earth = pv.PolyData(f"{examples_dir}/stl/earth_relief_10km.stl")
venus = pv.PolyData(f"{examples_dir}/stl/venus_relief_10km.stl")

# Apply textures to the planets
earth_texture = pv.Texture.from_path(f"{examples_dir}/textures/earthmap1k.jpg")
venus_texture = pv.Texture.from_path(f"{examples_dir}/textures/venusmap1k.jpg")

earth['texture'] = earth_texture
venus['texture'] = venus_texture

# Position the planets in a 3D space
sun = pv.Sphere(radius=1.0)
earth.translate((10, 0, 0))
venus.translate((5, 0, 0))

# Create a light source to simulate the sun
light_source = pv.Light(color=(1, 1, 0.5), intensity=10)

# Create a plotter and add the models to it
plotter = pv.Plotter()
plotter.add_light(light_source)
plotter.add_mesh(sun, color='yellow', show_edges=True)
plotter.add_mesh(earth, texture_scale=(1, 1), specular=0.2)
plotter.add_mesh(venus, texture_scale=(1, 1), specular=0.2)

# Create subplots for individual planets, showing their textures
plotter.subplot(earth, title="Earth Texture")
plotter.subplot(venus, title="Venus Texture")

# Create a visualization of Venus with and without its atmosphere
venus_with_atmosphere = pv.PolyData(f"{examples_dir}/stl/venus_relief_10km.stl")
venus_without_atmosphere = venus.copy()
venus_without_atmosphere.points[:, 2] += 0.05

plotter.add_mesh(venus_with_atmosphere, texture_scale=(1, 1), specular=0.2, opacity=0.5, name="Venus with Atmosphere")
plotter.add_mesh(venus_without_atmosphere, texture_scale=(1, 1), specular=0.2, name="Venus without Atmosphere")
plotter.legend(ncol=2)

# Display the plot
plotter.show()