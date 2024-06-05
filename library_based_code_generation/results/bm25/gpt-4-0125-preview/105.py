```python
import pyvista as pv
from pyvista import examples

# Create a plotting object
plotter = pv.Plotter()

# Supertoroid
supertoroid = examples.parametric_objects.ParametricSuperToroid()
plotter.add_mesh(supertoroid, color='lightblue')

# Ellipsoid
ellipsoid = examples.parametric_objects.ParametricEllipsoid()
plotter.add_mesh(ellipsoid, color='lightblue')

# Partial Parametric Ellipsoid
partial_ellipsoid = examples.parametric_objects.ParametricEllipsoid(u_max=np.pi/2, v_max=np.pi/2)
plotter.add_mesh(partial_ellipsoid, color='lightblue')

# Pseudosphere
pseudosphere = examples.parametric_objects.ParametricPseudosphere()
plotter.add_mesh(pseudosphere, color='lightblue')

# Bohemian Dome
bohemian_dome = examples.parametric_objects.ParametricBohemianDome()
plotter.add_mesh(bohemian_dome, color='lightblue')

# Bour
bour = examples.parametric_objects.ParametricBour()
plotter.add_mesh(bour, color='lightblue')

# Boy's Surface
boy_surface = examples.parametric_objects.ParametricBoy()
plotter.add_mesh(boy_surface, color='lightblue')

# Catalan Minimal
catalan_minimal = examples.parametric_objects.ParametricCatalanMinimal()
plotter.add_mesh(catalan_minimal, color='lightblue')

# Conic Spiral
conic_spiral = examples.parametric_objects.ParametricConicSpiral()
plotter.add_mesh(conic_spiral, color='lightblue')

# Cross Cap
cross_cap = examples.parametric_objects.ParametricCrossCap()
plotter.add_mesh(cross_cap, color='lightblue')

# Dini
dini = examples.parametric_objects.ParametricDini()
plotter.add_mesh(dini, color='lightblue')

# Enneper
enneper = examples.parametric_objects.ParametricEnneper()
plotter.add_mesh(enneper, color='lightblue', position='yz')

# Figure-8 Klein
figure_8_klein = examples.parametric_objects.ParametricFigure8Klein()
plotter.add_mesh(figure_8_klein, color='lightblue')

# Henneberg
henneberg = examples.parametric_objects.ParametricHenneberg()
plotter.add_mesh(henneberg, color='lightblue')

# Klein
klein = examples.parametric_objects.ParametricKlein()
plotter.add_mesh(klein, color='lightblue')

# Kuen
kuen = examples.parametric_objects.ParametricKuen()
plotter.add_mesh(kuen, color='lightblue')

# Mobius
mobius = examples.parametric_objects.ParametricMobius()
plotter.add_mesh(mobius, color='lightblue')

# Plucker Conoid
plucker_conoid = examples.parametric_objects.ParametricPluckerConoid()
plotter.add_mesh(plucker_conoid, color='lightblue')

# Random Hills
random_hills = examples.parametric_objects.ParametricRandomHills()
plotter.add_mesh(random_hills, color='lightblue')

# Roman
roman = examples.parametric_objects.ParametricRoman()
plotter.add_mesh(roman, color='lightblue')

# Super Ellipsoid
super_ellipsoid = examples.parametric_objects.ParametricSuperEllipsoid()
plotter.add_mesh(super_ellipsoid, color='lightblue')

# Torus
torus = examples.parametric_objects.ParametricTorus()
plotter.add_mesh(torus, color='lightblue')

# Circular Arc
start_point = [1, 0, 0]
end_point = [-1, 0, 0]
center_point = [0, 0, 0]
circular_arc = pv.CircularArc(start_point, end_point, center_point)
plotter.add_mesh(circular_arc, color='lightblue')

# Extruded Half Arc
half_arc = pv.CircularArcFromNormal(center=center_point, resolution=100, normal=[0, 0, 1], polar=[0, 1, 0], angle=180)
extruded_half_arc = half_arc.extrude([0, 0, 1])
plotter.add_mesh(extruded_half_arc, color='lightblue', show_edges=True)

# Display all the objects
plotter.show()
```