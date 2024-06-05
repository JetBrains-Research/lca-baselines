  ```
from pymapdl import launch_mapdl

# Launch MAPDL and enter the pre-processor
mapdl = launch_mapdl()
mapdl.prep7()

# Create a block and mesh it with tetrahedral elements
mapdl.block(0, 1, 0, 1, 0, 1)
mapdl.et(1, 100)
mapdl.emodif(1, 1)
mapdl.allsel()
mapdl.mesh()

# Create a second volume block above the existing one and mesh it with quadratic hexahedral elements
mapdl.block(0, 1, 0, 1, 0, 1, 1)
mapdl.et(2, 101)
mapdl.emodif(2, 1)
mapdl.allsel()
mapdl.mesh()

# Select all the elements at the intersection between the two blocks
mapdl.select_elements(1, 2)

# Generate contact elements
mapdl.contact_geom_and_mesh(1, 2)

# Plot the contact element pairs as a wire-frame to show that the contact pairs overlap
mapdl.plot_nodal_contact_friction_stress(1, 2)

# Stop MAPDL
mapdl.quit()
```