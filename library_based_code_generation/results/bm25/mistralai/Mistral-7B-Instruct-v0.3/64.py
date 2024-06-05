 Here is a basic example of how you might structure your code using the pymapdl library. Please note that this is a simplified version and you may need to adjust it according to your specific needs.

```python
from pymapdl.mapdl_interface import MapdlInterface
import numpy as np

# Launch MAPDL and set units to US Customary system using inches
mapdl = MapdlInterface()
mapdl.test_raise_exec_path_and_version_launcher()
mapdl.run_before_and_after_tests('INCH', 'FPS')

# Define materials and element type
mapdl.test_set_using_a_component('MATERIAL', '7-Aluminum Alloy 6061-T6')
mapdl.test_set_using_a_component('ELEMENT', '4-Quadrilateral Isoparametric')

# Create the geometry of the pressure vessel
geometry = mapdl.create_geometry()
mapdl._create_geometry('CUBE', 2, 2, 2, 0, 0, 0, 0, 0, 0)

# Create a mesh
mapdl.cube_geom_and_mesh(1, 10, 10, 10)

# Apply boundary conditions and pressure
mapdl.set_boundary_conditions('ALL', 'FIXED', 'U', 'X', 'Y', 'Z')
mapdl.set_boundary_conditions('1', 'PRESSURE', 'P', 100000)

# Solve the problem
mapdl.run_analysis('STATIC', 'NORMAL')

# Post-process the results to obtain the von-mises stress for the single static solution
results = mapdl.Results()
von_mises_stress = results.von_mises_stress()

# Plot the results and compare them with the results obtained from the legacy file reader
mapdl.plot_nodal_pressure('VON MISES STRESS', 'VON MISES')

# Stop MAPDL
mapdl.end()
```

This code creates a simple pressure vessel, solves it using static analysis, and plots the von-mises stress. However, it does not include the comparison with the results obtained from a legacy file reader, as that would require additional functionality not provided by the pymapdl library. You would need to implement that part separately.