 Here is a Python script that creates and plots the requested geometric objects using the pyvista library. Please note that this script assumes you have pyvista and numpy installed.

```python
import numpy as np
import pyvista

# Supertoroid
supertoroid = pyvista.SuperToroid(inner_radius=1, outer_radius=2)
supertoroid.plot(color='lightblue')

# Ellipsoid
ellipsoid = pyvista.Ellipsoid(major_axis=2, minor_axis=1, slices=10, stacks=10)
ellipsoid.plot(color='lightblue')

# Partial Parametric Ellipsoid (plotting direction: x)
partial_ellipsoid = pyvista.PartialParametricEllipsoid(major_axis=2, minor_axis=1, slices=10, stacks=10, direction='x')
partial_ellipsoid.plot(color='lightblue')

# Pseudosphere
pseudosphere = pyvista.Pseudosphere(radius=1, u_slices=10, v_slices=10)
pseudosphere.plot(color='lightblue')

# Bohemian Dome
bohemian_dome = pyvista.BohemianDome(radius=1, height=1, u_slices=10, v_slices=10)
bohemian_dome.plot(color='lightblue')

# Bour
bour = pyvista.Bour(radius=1, height=1, u_slices=10, v_slices=10)
bour.plot(color='lightblue')

# Boy's Surface
boys_surface = pyvista.BoySurface(radius=1, height=1, u_slices=10, v_slices=10)
boys_surface.plot(color='lightblue')

# Catalan Minimal
catalan_minimal = pyvista.CatalanMinimal(radius=1, height=1, u_slices=10, v_slices=10)
catalan_minimal.plot(color='lightblue')

# Conic Spiral
conic_spiral = pyvista.ConicSpiral(start_angle=0, end_angle=np.pi*2, start_radius=1, end_radius=2, slices=100)
conic_spiral.plot(color='lightblue')

# Cross Cap
cross_cap = pyvista.CrossCap(radius=1, height=1, u_slices=10, v_slices=10)
cross_cap.plot(color='lightblue')

# Dini
dini = pyvista.Dini(radius=1, height=1, u_slices=10, v_slices=10)
dini.plot(color='lightblue')

# Enneper (plotting position: yz)
enneper = pyvista.Enneper(radius=1, height=1, u_slices=10, v_slices=10)
enneper.plot_position = 'yz'
enneper.plot(color='lightblue')

# Figure-8 Klein
figure_8_klein = pyvista.Figure8Klein(radius=1, height=1, u_slices=10, v_slices=10)
figure_8_klein.plot(color='lightblue')

# Henneberg
henneberg = pyvista.Henneberg(radius=1, height=1, u_slices=10, v_slices=10)
henneberg.plot(color='lightblue')

# Klein
klein = pyvista.Klein(radius=1, height=1, u_slices=10, v_slices=10)
klein.plot(color='lightblue')

# Kuen
kuen = pyvista.Kuen(radius=1, height=1, u_slices=10, v_slices=10)
kuen.plot(color='lightblue')

# Mobius
mobius = pyvista.Mobius(radius=1, height=1, u_slices=10, v_slices=10)
mobius.plot(color='lightblue')

# Plucker Conoid
plucker_conoid = pyvista.PluckerConoid(radius=1, height=1, u_slices=10, v_slices=10)
plucker_conoid.plot(color='lightblue')

# Random Hills
random_hills = pyvista.RandomHills(height=1, width=1, u_slices=10, v_slices=10)
random_hills.plot(color='lightblue')

# Roman
roman = pyvista.Roman(radius=1, height=1, u_slices=10, v_slices=10)
roman.plot(color='lightblue')

# Super Ellipsoid
super_ellipsoid = pyvista.SuperEllipsoid(major_axis=2, minor_axis=1, slices=10, stacks=10)
super_ellipsoid.plot(color='lightblue')

# Torus
torus = pyvista.Torus(radius=1, tube_radius=0.5, u_slices=10, v_slices=10)
torus.plot(color='lightblue')

# Circular Arc
center = [0, 0, 0]
points = np.array([[1, 0, 0], [0, 1, 0]])
circular_arc = pyvista.PolyData(points)
circular_arc.plot(center=center, color='lightblue')

# Extruded Half Arc
center = [0, 0, 0]
points = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
extruded_half_arc = pyvista.PolyData(points)
extruded_half_arc.extrude(extent=(0, 1), vectors=np.array([0, 0, 1]))
extruded_half_arc.edges.plot(color='black')
extruded_half_arc.plot(center=center, color='lightblue')
```

This script creates each geometric object and plots it with light blue color where applicable. For the Partial Parametric Ellipsoid, the plotting direction is set to 'x'. For the Enneper, the plotting position is set to 'yz'. For the Circular Arc and the Extruded Half Arc, specific points and a center are defined. The Extruded Half Arc is extruded in the z direction, and its edges are shown in the plot.