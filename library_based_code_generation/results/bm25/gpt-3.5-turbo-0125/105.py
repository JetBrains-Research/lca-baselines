import pyvista as pv

# Create and plot Supertoroid
supertoroid = pv.ParametricSuperEllipsoid()
plot_supertoroid = pv.Plotter()
plot_supertoroid.add_mesh(supertoroid, color='lightblue')
plot_supertoroid.show()

# Create and plot Ellipsoid
ellipsoid = pv.ParametricEllipsoid()
plot_ellipsoid = pv.Plotter()
plot_ellipsoid.add_mesh(ellipsoid, color='lightblue')
plot_ellipsoid.show()

# Create and plot Partial Parametric Ellipsoid
partial_ellipsoid = pv.ParametricEllipsoid()
plot_partial_ellipsoid = pv.Plotter()
plot_partial_ellipsoid.add_mesh(partial_ellipsoid, direction=[1, 1, 1], color='lightblue')
plot_partial_ellipsoid.show()

# Create and plot Pseudosphere
pseudosphere = pv.ParametricEnneper()
plot_pseudosphere = pv.Plotter()
plot_pseudosphere.add_mesh(pseudosphere, color='lightblue')
plot_pseudosphere.show()

# Create and plot Bohemian Dome
bohemian_dome = pv.ParametricKuen()
plot_bohemian_dome = pv.Plotter()
plot_bohemian_dome.add_mesh(bohemian_dome, color='lightblue')
plot_bohemian_dome.show()

# Create and plot Bour
bour = pv.ParametricKlein()
plot_bour = pv.Plotter()
plot_bour.add_mesh(bour, color='lightblue')
plot_bour.show()

# Create and plot Boy's Surface
boys_surface = pv.ParametricEnneper()
plot_boys_surface = pv.Plotter()
plot_boys_surface.add_mesh(boys_surface, position='yz', color='lightblue')
plot_boys_surface.show()

# Create and plot Catalan Minimal
catalan_minimal = pv.ParametricKuen()
plot_catalan_minimal = pv.Plotter()
plot_catalan_minimal.add_mesh(catalan_minimal, color='lightblue')
plot_catalan_minimal.show()

# Create and plot Conic Spiral
conic_spiral = pv.CircularArc()
plot_conic_spiral = pv.Plotter()
plot_conic_spiral.add_mesh(conic_spiral, color='lightblue')
plot_conic_spiral.show()

# Create and plot Cross Cap
cross_cap = pv.ParametricEnneper()
plot_cross_cap = pv.Plotter()
plot_cross_cap.add_mesh(cross_cap, color='lightblue')
plot_cross_cap.show()

# Create and plot Dini
dini = pv.ParametricEnneper()
plot_dini = pv.Plotter()
plot_dini.add_mesh(dini, color='lightblue')
plot_dini.show()

# Create and plot Enneper
enneper = pv.ParametricEnneper()
plot_enneper = pv.Plotter()
plot_enneper.add_mesh(enneper, position='yz', color='lightblue')
plot_enneper.show()

# Create and plot Figure-8 Klein
figure8_klein = pv.ParametricKuen()
plot_figure8_klein = pv.Plotter()
plot_figure8_klein.add_mesh(figure8_klein, color='lightblue')
plot_figure8_klein.show()

# Create and plot Henneberg
henneberg = pv.ParametricKuen()
plot_henneberg = pv.Plotter()
plot_henneberg.add_mesh(henneberg, color='lightblue')
plot_henneberg.show()

# Create and plot Klein
klein = pv.ParametricKuen()
plot_klein = pv.Plotter()
plot_klein.add_mesh(klein, color='lightblue')
plot_klein.show()

# Create and plot Kuen
kuen = pv.ParametricKuen()
plot_kuen = pv.Plotter()
plot_kuen.add_mesh(kuen, color='lightblue')
plot_kuen.show()

# Create and plot Mobius
mobius = pv.ParametricEnneper()
plot_mobius = pv.Plotter()
plot_mobius.add_mesh(mobius, color='lightblue')
plot_mobius.show()

# Create and plot Plucker Conoid
plucker_conoid = pv.ParametricKuen()
plot_plucker_conoid = pv.Plotter()
plot_plucker_conoid.add_mesh(plucker_conoid, color='lightblue')
plot_plucker_conoid.show()

# Create and plot Random Hills
random_hills = pv.ParametricKuen()
plot_random_hills = pv.Plotter()
plot_random_hills.add_mesh(random_hills, color='lightblue')
plot_random_hills.show()

# Create and plot Roman
roman = pv.ParametricKuen()
plot_roman = pv.Plotter()
plot_roman.add_mesh(roman, color='lightblue')
plot_roman.show()

# Create and plot Super Ellipsoid
super_ellipsoid = pv.ParametricSuperEllipsoid()
plot_super_ellipsoid = pv.Plotter()
plot_super_ellipsoid.add_mesh(super_ellipsoid, color='lightblue')
plot_super_ellipsoid.show()

# Create and plot Torus
torus = pv.ParametricSuperEllipsoid()
plot_torus = pv.Plotter()
plot_torus.add_mesh(torus, color='lightblue')
plot_torus.show()

# Create and plot Circular Arc
circular_arc = pv.CircularArc()
plot_circular_arc = pv.Plotter()
plot_circular_arc.add_mesh(circular_arc, color='lightblue')
plot_circular_arc.show()

# Create and plot Extruded Half Arc
extruded_half_arc = pv.CircularArc()
center = [0, 0, 0]
point1 = [1, 0, 0]
point2 = [0, 1, 0]
plot_extruded_half_arc = pv.Plotter()
plot_extruded_half_arc.add_mesh(extruded_half_arc.extrude([0, 0, 1]), color='lightblue', show_edges=True)
plot_extruded_half_arc.show()