import pyvista as pv

# Create and plot Supertoroid
supertoroid = pv.ParametricSuperToroid()
supertoroid.plot(color='lightblue')

# Create and plot Ellipsoid
ellipsoid = pv.ParametricEllipsoid()
ellipsoid.plot(color='lightblue')

# Create and plot Partial Parametric Ellipsoid
partial_ellipsoid = pv.ParametricEllipsoid()
partial_ellipsoid.plot(direction=(1, 1, 1), color='lightblue')

# Create and plot Pseudosphere
pseudosphere = pv.ParametricPseudosphere()
pseudosphere.plot(color='lightblue')

# Create and plot Bohemian Dome
bohemian_dome = pv.ParametricBohemianDome()
bohemian_dome.plot(color='lightblue')

# Create and plot Bour
bour = pv.ParametricBour()
bour.plot(color='lightblue')

# Create and plot Boy's Surface
boys_surface = pv.ParametricBoy()
boys_surface.plot(color='lightblue')

# Create and plot Catalan Minimal
catalan_minimal = pv.ParametricCatalanMinimal()
catalan_minimal.plot(color='lightblue')

# Create and plot Conic Spiral
conic_spiral = pv.ParametricConicSpiral()
conic_spiral.plot(color='lightblue')

# Create and plot Cross Cap
cross_cap = pv.ParametricCrossCap()
cross_cap.plot(color='lightblue')

# Create and plot Dini
dini = pv.ParametricDini()
dini.plot(color='lightblue')

# Create and plot Enneper
enneper = pv.ParametricEnneper()
enneper.plot(position='yz', color='lightblue')

# Create and plot Figure-8 Klein
figure8_klein = pv.ParametricFigure8Klein()
figure8_klein.plot(color='lightblue')

# Create and plot Henneberg
henneberg = pv.ParametricHenneberg()
henneberg.plot(color='lightblue')

# Create and plot Klein
klein = pv.ParametricKlein()
klein.plot(color='lightblue')

# Create and plot Kuen
kuen = pv.ParametricKuen()
kuen.plot(color='lightblue')

# Create and plot Mobius
mobius = pv.ParametricMobius()
mobius.plot(color='lightblue')

# Create and plot Plucker Conoid
plucker_conoid = pv.ParametricPluckerConoid()
plucker_conoid.plot(color='lightblue')

# Create and plot Random Hills
random_hills = pv.ParametricRandomHills()
random_hills.plot(color='lightblue')

# Create and plot Roman
roman = pv.ParametricRoman()
roman.plot(color='lightblue')

# Create and plot Super Ellipsoid
super_ellipsoid = pv.ParametricSuperEllipsoid()
super_ellipsoid.plot(color='lightblue')

# Create and plot Torus
torus = pv.ParametricTorus()
torus.plot(color='lightblue')

# Create and plot Circular Arc
circular_arc = pv.CircularArc(pointa=(0, 0, 0), pointb=(1, 1, 1), center=(0.5, 0.5, 0.5))
circular_arc.plot(color='lightblue')

# Create and plot Extruded Half Arc
extruded_half_arc = pv.ExtrudedHalfArc(pointa=(0, 0, 0), pointb=(1, 1, 1), center=(0.5, 0.5, 0.5), direction='z', show_edges=True)
extruded_half_arc.plot(color='lightblue')