  import pyvista as pv

# Supertoroid
supertoroid = pv.Supertoroid()
supertoroid.plot(color='lightblue')

# Ellipsoid
ellipsoid = pv.Ellipsoid()
ellipsoid.plot(color='lightblue')

# Partial Parametric Ellipsoid
partial_ellipsoid = pv.PartialParametricEllipsoid(direction='yz')
partial_ellipsoid.plot(color='lightblue')

# Pseudosphere
pseudosphere = pv.Pseudosphere()
pseudosphere.plot(color='lightblue')

# Bohemian Dome
bohemian_dome = pv.BohemianDome()
bohemian_dome.plot(color='lightblue')

# Bour
bour = pv.Bour()
bour.plot(color='lightblue')

# Boy's Surface
boys_surface = pv.BoySurface()
boys_surface.plot(color='lightblue')

# Catalan Minimal
catalan_minimal = pv.CatalanMinimal()
catalan_minimal.plot(color='lightblue')

# Conic Spiral
conic_spiral = pv.ConicSpiral()
conic_spiral.plot(color='lightblue')

# Cross Cap
cross_cap = pv.CrossCap()
cross_cap.plot(color='lightblue')

# Dini
dini = pv.Dini()
dini.plot(color='lightblue')

# Enneper
enneper = pv.Enneper()
enneper.plot(color='lightblue', position='yz')

# Figure-8 Klein
figure_8_klein = pv.Figure8Klein()
figure_8_klein.plot(color='lightblue')

# Henneberg
henneberg = pv.Henneberg()
henneberg.plot(color='lightblue')

# Klein
klein = pv.Klein()
klein.plot(color='lightblue')

# Kuen
kuen = pv.Kuen()
kuen.plot(color='lightblue')

# Mobius
mobius = pv.Mobius()
mobius.plot(color='lightblue')

# Plucker Conoid
plucker_conoid = pv.PluckerConoid()
plucker_conoid.plot(color='lightblue')

# Random Hills
random_hills = pv.RandomHills()
random_hills.plot(color='lightblue')

# Roman
roman = pv.Roman()
roman.plot(color='lightblue')

# Super Ellipsoid
super_ellipsoid = pv.SuperEllipsoid()
super_ellipsoid.plot(color='lightblue')

# Torus
torus = pv.Torus()
torus.plot(color='lightblue')

# Circular Arc
circular_arc = pv.CircularArc(center=(0, 0, 0), radius=1, theta1=0, theta2=360)
circular_arc.plot(color='lightblue')

# Extruded Half Arc
extruded_half_arc = pv.ExtrudedHalfArc(center=(0, 0, 0), radius=1, theta1=0, theta2=360, direction='z')
extruded_half_arc.plot(color='lightblue', show_edges=True)