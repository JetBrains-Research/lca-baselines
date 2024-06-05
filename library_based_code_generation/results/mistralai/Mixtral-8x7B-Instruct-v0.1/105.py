 import pyvista as pv
import numpy as np

# Supertoroid
supertoroid = pv.ParametricSuperToroid(radius_outer=1, radius_inner=0.5, height=1, rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(supertoroid, color='lightblue')
plotter.show()

# Ellipsoid
ellipsoid = pv.ParametricEllipsoid(radius_x=1, radius_y=0.5, radius_z=0.25)
plotter = pv.Plotter()
plotter.add_mesh(ellipsoid, color='lightblue')
plotter.show()

# Partial Parametric Ellipsoid
partial_ellipsoid = pv.ParametricPartialEllipsoid(radius_x=1, radius_y=0.5, radius_z=0.25, start_phi=0, end_phi=np.pi/2)
plotter = pv.Plotter()
plotter.add_mesh(partial_ellipsoid, color='lightblue')
plotter.show()

# Pseudosphere
pseudosphere = pv.ParametricPseudosphere(radius=1, height=1, rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(pseudosphere, color='lightblue')
plotter.show()

# Bohemian Dome
bohemian_dome = pv.ParametricBohemianDome(radius=1, height=1, rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(bohemian_dome, color='lightblue')
plotter.show()

# Bour
bour = pv.ParametricBour(radius=1, height=1, rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(bour, color='lightblue')
plotter.show()

# Boy's Surface
boys_surface = pv.ParametricBoySurface(rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(boys_surface, color='lightblue')
plotter.show()

# Catalan Minimal
catalan_minimal = pv.ParametricCatalanMinimal(rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(catalan_minimal, color='lightblue')
plotter.show()

# Conic Spiral
conic_spiral = pv.ParametricConicSpiral(a=1, b=0.5, c=0.25, d=1, k=1, start_theta=0, end_theta=2*np.pi)
plotter = pv.Plotter()
plotter.add_mesh(conic_spiral, color='lightblue')
plotter.show()

# Cross Cap
cross_cap = pv.ParametricCrossCap(radius=1, height=1, rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(cross_cap, color='lightblue')
plotter.show()

# Dini
dini = pv.ParametricDini(a=1, b=0.5, c=0.25, start_u=0, end_u=2*np.pi, start_v=0, end_v=np.pi)
plotter = pv.Plotter()
plotter.add_mesh(dini, color='lightblue')
plotter.show()

# Enneper
enneper = pv.ParametricEnneper(rings=30, circles=30)
plotter = pv.Plotter(window_size=[500, 500], off_screen=True, plot_position='yz')
plotter.add_mesh(enneper, color='lightblue')
plotter.show()

# Figure-8 Klein
figure_8_klein = pv.ParametricFigure8Klein(rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(figure_8_klein, color='lightblue')
plotter.show()

# Henneberg
henneberg = pv.ParametricHenneberg(rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(henneberg, color='lightblue')
plotter.show()

# Klein
klein = pv.ParametricKlein(rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(klein, color='lightblue')
plotter.show()

# Kuen
kuen = pv.ParametricKuen(rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(kuen, color='lightblue')
plotter.show()

# Mobius
mobius = pv.ParametricMobius(rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(mobius, color='lightblue')
plotter.show()

# Plucker Conoid
plucker_conoid = pv.ParametricPluckerConoid(a=1, b=0.5, c=0.25, start_u=0, end_u=2*np.pi, start_v=0, end_v=np.pi)
plotter = pv.Plotter()
plotter.add_mesh(plucker_conoid, color='lightblue')
plotter.show()

# Random Hills
random_hills = pv.ParametricRandomHills(dimensions=10, height=10, n_points=100000)
plotter = pv.Plotter()
plotter.add_mesh(random_hills, color='lightblue')
plotter.show()

# Super Ellipsoid
super_ellipsoid = pv.ParametricSuperEllipsoid(radius=1, exponent=2, height=1, rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(super_ellipsoid, color='lightblue')
plotter.show()

# Torus
torus = pv.ParametricTorus(radius=1, tube=0.5, rings=30, circles=30)
plotter = pv.Plotter()
plotter.add_mesh(torus, color='lightblue')
plotter.show()

# Circular Arc
circular_arc = pv.ParametricCircularArc(point1=[0, 0, 0], point2=[1, 0, 0], center=[0.5, 1, 0])
plotter = pv.Plotter()
plotter.add_mesh(circular_arc, color='lightblue')
plotter.show()

# Extruded Half Arc
half_arc = pv.ParametricCircularArc(point1=[0, 0, 0], point2=[1, 0, 0], center=[0.5, 1, 0])
extruded_half_arc = half_arc.extrude(straight=1)
plotter = pv.Plotter()
plotter.add_mesh(extruded_half_arc, color='lightblue', line_width=5)
plotter.show()