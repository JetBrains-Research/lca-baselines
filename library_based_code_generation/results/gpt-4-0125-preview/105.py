```python
import pyvista as pv

# Create a plotting window
plotter = pv.Plotter()

# Define specific points and center for Circular Arc and Extruded Half Arc
points = [[1, 0, 0], [0, 1, 0], [-1, 0, 0]]
center = [0, 0, 0]

# List of parametric objects to create and plot
parametric_objects = [
    pv.ParametricSupertoroid(),
    pv.ParametricEllipsoid(),
    pv.ParametricPartialEllipsoid(phi_resolution=10, theta_resolution=10, start_phi=0, end_phi=180, start_theta=0, end_theta=180),
    pv.ParametricPseudosphere(),
    pv.ParametricBohemianDome(),
    pv.ParametricBour(),
    pv.ParametricBoy(),
    pv.ParametricCatalanMinimal(),
    pv.ParametricConicSpiral(),
    pv.ParametricCrossCap(),
    pv.ParametricDini(),
    pv.ParametricEnneper(),
    pv.ParametricFigure8Klein(),
    pv.ParametricHenneberg(),
    pv.ParametricKlein(),
    pv.ParametricKuen(),
    pv.ParametricMobius(),
    pv.ParametricPluckerConoid(),
    pv.ParametricRandomHills(),
    pv.ParametricRoman(),
    pv.ParametricSuperEllipsoid(),
    pv.ParametricTorus(),
    pv.ParametricCircularArc(points[0], points[1], points[2]),
    pv.ParametricExtrudedHalfArc(center, points[0], points[2], extrusion_direction=[0, 0, 1])
]

# Plot each object
for obj in parametric_objects:
    if isinstance(obj, pv.ParametricEnneper):
        plotter.add_mesh(obj, color='lightblue', position='yz')
    else:
        plotter.add_mesh(obj, color='lightblue')

# Show edges for the Extruded Half Arc
plotter.add_mesh(parametric_objects[-1], show_edges=True)

# Display the plot
plotter.show()
```