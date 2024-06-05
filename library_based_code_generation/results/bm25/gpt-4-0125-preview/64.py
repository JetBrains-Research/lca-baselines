```python
from ansys.mapdl.core import launch_mapdl

# Launch MAPDL and set units to US Customary (inches)
mapdl = launch_mapdl()
mapdl.prep7()
mapdl.units('BIN')

# Define materials
mapdl.mp('EX', 1, 30e6)  # Young's modulus in psi
mapdl.mp('NUXY', 1, 0.3)  # Poisson's ratio
mapdl.mp('DENS', 1, 0.284)  # Density in lb/in^3

# Define element type
mapdl.et(1, 'SOLID186')  # 3D 20-node structural solid

# Create geometry
mapdl.block(0, 10, 0, 10, 0, 20)  # Create a block with dimensions 10x10x20 inches

# Create mesh
mapdl.esize(1)  # Element size
mapdl.vmesh('ALL')

# Apply boundary conditions
mapdl.nsel('S', 'LOC', 'Z', 0)
mapdl.d('ALL', 'ALL')
mapdl.nsel('ALL')

# Apply pressure
mapdl.nsel('S', 'LOC', 'Z', 20)
mapdl.sf('ALL', 'PRES', 100)  # Apply 100 psi pressure on the top face
mapdl.nsel('ALL')

# Solve
mapdl.solve()

# Post-processing
mapdl.post1()  # Enter post-processing mode
mapdl.set(1)  # Select the first (and only) set of results

# Obtain the von-Mises stress
mapdl.post_processing.plot_nodal_stress('SEQV')

# Compare results with legacy file reader
# Assuming the existence of a function to read legacy results and a variable `legacy_results`
# legacy_results = read_legacy_results('legacy_file.rst')
# current_results = mapdl.result.nodal_stress(0)
# comparison = are_the_same_result(current_results, legacy_results)
# print(f"Comparison with legacy results: {comparison}")

# Stop MAPDL
mapdl.exit()
```