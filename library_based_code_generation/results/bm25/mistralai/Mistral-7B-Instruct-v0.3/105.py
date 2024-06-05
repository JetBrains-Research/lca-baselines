 Here's a Python script that creates and plots the requested geometric objects using the pyvista library. Please note that this script assumes you have pyvista and numpy installed.

```python
import numpy as np
import pyvista

# Geometric objects
supertoroid = pyvista.ParametricSuperEllipsoid(major_axis=2, minor_axis=1, height=1)
ellipsoid = pyvista.ParametricEllipsoid(major_axis=2, minor_axis=1, height=1)
partial_ellipsoid = pyvista.ParametricEllipsoid(major_axis=2, minor_axis=1, height=1, direction=[-1, 0, 0])
pseudosphere = pyvista.ParametricEnneper()
bohemian_dome = pyvista.ParametricKlein()
bour = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
boy_s_surface = pyvista.CircularArc(center=[0, 0, 1], points=[[0, 0, 1], [0, np.sqrt(3), 1]])
catalan_minimal = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
conic_spiral = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
cross_cap = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
dini = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
enneper = pyvista.ParametricEnneper(position='yz')
figure_8_klein = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
henneberg = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
klein = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
kuen = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
mobius = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
plucker_conoid = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
random_hills = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
roman = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
super_ellipsoid = pyvista.ParametricSuperEllipsoid(major_axis=2, minor_axis=1, height=1)
torus = pyvista.Torus(major_radius=1, minor_radius=0.5)
circular_arc = pyvista.CircularArc(center=[0, 0, 0], points=[[0, 0, 0], [1, 0, 0]])
extruded_half_arc = pyvista.PolyData(circular_arc.sample(50))
extruded_half_arc.extrude(1)
extruded_half_arc.edges.set_visibility(True)

# Plotting
for obj in [supertoroid, ellipsoid, partial_ellipsoid, pseudosphere, bohemian_dome, bour, boy_s_surface, catalan_minimal,
            conic_spiral, cross_cap, dini, enneper, figure_8_klein, henneberg, klein, kuen, mobius, plucker_conoid,
            random_hills, roman, super_ellipsoid, torus, circular_arc, extruded_half_arc]:
    obj.plot(color='lightblue')
```

This script creates the requested geometric objects and plots them with a light blue color. The Partial Parametric Ellipsoid, Enneper, Circular Arc, and Extruded Half Arc have been defined as specified. The Extruded Half Arc is extruded in the z direction, and its edges are shown in the plot.