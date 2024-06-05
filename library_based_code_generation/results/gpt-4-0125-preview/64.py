```python
from ansys.mapdl.core import launch_mapdl

# Launch MAPDL
mapdl = launch_mapdl()

# Set units to US Customary system (inches)
mapdl.prep7()
mapdl.units('BIN')

# Define materials
mapdl.mp('EX', 1, 30e6)  # Young's modulus in psi
mapdl.mp('NUXY', 1, 0.3)  # Poisson's Ratio
mapdl.mp('DENS', 1, 0.284)  # Density in lb/in^3

# Define element type
mapdl.et(1, 'SOLID186')  # 3D 20-node structural solid

# Create geometry
mapdl.block(0, 10, 0, 10, 0, 10)  # Create a block with dimensions 10x10x10 inches

# Create mesh
mapdl.esize(1)  # Element size
mapdl.vmesh('ALL')  # Mesh the volume

# Apply boundary conditions
mapdl.nsel('S', 'LOC', 'Z', 0)  # Select nodes at Z=0
mapdl.d('ALL', 'ALL')  # Constrain all DOFs for these nodes

# Apply pressure
mapdl.nsel('S', 'LOC', 'Z', 10)  # Select nodes at Z=10
mapdl.sf('ALL', 'PRES', 100)  # Apply 100 psi pressure
mapdl.allsel()  # Select all nodes

# Solve the problem
mapdl.run('/SOLU')
mapdl.antype('STATIC')
mapdl.solve()
mapdl.finish()

# Post-processing
mapdl.post1()  # Enter post-processing
mapdl.set(1)  # Select the first (and only) set of results
mapdl.post_processing.plot_nodal_stress('SEQV')  # Plot von Mises stress

# Compare with results obtained from the legacy file reader
mapdl.post_processing.plot_nodal_solution('U', 'X')  # Plot X displacement

# Stop MAPDL
mapdl.exit()
```