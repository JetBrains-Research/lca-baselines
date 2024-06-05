  Here is the code to create and plot the various parametric geometric objects using the pyvista library:
```
import pyvista as pv

# Create a Supertoroid
supertoroid = pv.ParametricSuperEllipsoid(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create an Ellipsoid
ellipsoid = pv.ParametricEllipsoid(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Partial Parametric Ellipsoid
partial_ellipsoid = pv.ParametricEllipsoid(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, plot_direction='yz')

# Create a Pseudosphere
pseudosphere = pv.ParametricPseudosphere(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Bohemian Dome
bohemian_dome = pv.ParametricBohemianDome(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Bour
bour = pv.ParametricBour(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Boy's Surface
boys_surface = pv.ParametricBoySurface(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Catalan Minimal
catalan_minimal = pv.ParametricCatalanMinimal(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Conic Spiral
conic_spiral = pv.ParametricConicSpiral(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Cross Cap
cross_cap = pv.ParametricCrossCap(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Dini
dini = pv.ParametricDini(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create an Enneper
enneper = pv.ParametricEnneper(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, plot_position='yz')

# Create a Figure-8 Klein
figure_8_klein = pv.ParametricFigure8Klein(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Henneberg
henneberg = pv.ParametricHenneberg(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Klein
klein = pv.ParametricKlein(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Kuen
kuen = pv.ParametricKuen(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Mobius
mobius = pv.ParametricMobius(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Plucker Conoid
plucker_conoid = pv.ParametricPluckerConoid(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create Random Hills
random_hills = pv.ParametricRandomHills(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Roman
roman = pv.ParametricRoman(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Super Ellipsoid
super_ellipsoid = pv.ParametricSuperEllipsoid(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Torus
torus = pv.ParametricTorus(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create a Circular Arc
circular_arc = pv.ParametricCircularArc(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# Create an Extruded Half Arc
extruded_half_arc = pv.ParametricExtrudedHalfArc(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, center=(0, 0, 0), direction='z')

# Plot the objects
pv.plot(supertoroid, ellipsoid, partial_ellipsoid, pseudosphere, bohemian_dome, bour, boys_surface, catalan_minimal, conic_spiral, cross_cap, dini, enneper, figure_8_klein, henneberg, klein, kuen, mobius, plucker_conoid, random_hills, roman, super_ellipsoid, torus, circular_arc, extruded_half_arc, color='lightblue')
```
Note that the `Parametric*` classes in pyvista are used to create the parametric geometric objects, and the `plot()` function is used to plot them. The `color` argument is used to set the color of the objects to light blue.