  from pymapdl import *

# Launch MAPDL and enter the pre-processor
mapdl = launch_mapdl()
mapdl.prep7()

# Create a block and mesh it with tetrahedral elements
mapdl.et(1, "TET")
mapdl.block(0, 1, 0, 1, 0, 1)
mapdl.esize(0.1)
mapdl.emesh(0)

# Create a second volume block above the existing one and mesh it with quadratic hexahedral elements
mapdl.et(2, "HEX")
mapdl.block(1, 1, 0, 1, 0, 1)
mapdl.esize(0.1)
mapdl.emesh(1)

# Select all the elements at the intersection between the two blocks and generate contact elements
mapdl.secsel("S", "BLOCK", 0)
mapdl.secsel("S", "BLOCK", 1)
mapdl.contact("GENERAL")

# Plot the contact element pairs as a wire-frame to show that the contact pairs overlap
mapdl.vplot(show_wireframe=True)

# Stop MAPDL
mapdl.exit()