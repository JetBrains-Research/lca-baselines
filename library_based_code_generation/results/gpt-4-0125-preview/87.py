```python
from ansys.mapdl.core import launch_mapdl

# Launch MAPDL
mapdl = launch_mapdl()

# Enter pre-processor
mapdl.prep7()

# Create a block and mesh it with tetrahedral elements
mapdl.block(0, 10, 0, 10, 0, 5)
mapdl.et(1, "SOLID187")
mapdl.vmesh("ALL")

# Create a second volume block above the existing one
mapdl.voffst(6, 10, 10, 15, 1)
mapdl.et(2, "SOLID186")
mapdl.vmesh("ALL")

# Select all elements at the intersection between the two blocks
# Note: This is a simplified approach for demonstration. Actual intersection and contact pair generation may require more specific commands based on geometry and requirements.
mapdl.asel('S', 'VOLU', '', 1, 2)
mapdl.nsel('S', 'LOC', 'Z', 5)
mapdl.cm('CONTACT_NODES', 'NODE')
mapdl.esln('S', 1)
mapdl.cm('TARGET_ELEMS', 'ELEM')

# Generate contact elements
mapdl.et(3, "CONTA174")
mapdl.et(4, "TARGE170")
mapdl.type(3)
mapdl.real(3)
mapdl.esurf()
mapdl.type(4)
mapdl.real(4)
mapdl.esln('S', 1)
mapdl.esurf()

# Plot the contact element pairs as a wire-frame
mapdl.eplot(show_node_numbering=True, line_width=1, background='w', show_edges=True, edge_color='black', cpos='iso')

# Stop MAPDL
mapdl.exit()
```