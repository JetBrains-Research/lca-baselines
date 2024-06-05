 Here is a basic example of how you might structure your code using the pymapdl library. Please note that this is a simplified version and might need adjustments based on your specific pressure vessel geometry and requirements.

```python
import pymapdl

# Launch MAPDL and set units to US Customary system using inches
pymapdl.mapdl('init')
pymapdl.mapdl('units inch')

# Define materials
pymapdl.mapdl('define material 1 aluminum')
pymapdl.mapdl('property 1 elastic modulus 10e6')
pymapdl.mapdl('property 1 poisson ratio 0.33')
pymapdl.mapdl('property 1 density 0.097')

# Define element type
pymapdl.mapdl('definesolid185')

# Create geometry
# (Geometry creation code will depend on your specific pressure vessel geometry)

# Create mesh
# (Mesh creation code will depend on your specific pressure vessel geometry)

# Apply boundary conditions
# (Boundary condition application code will depend on your specific pressure vessel)

# Apply pressure
pymapdl.mapdl('load pressure')
pymapdl.mapdl('point 0,0,0')
pymapdl.mapdl('value 1000')

# Solve the problem
pymapdl.mapdl('solve static')

# Post-process results
pymapdl.mapdl('plot stress von-mises')

# Read legacy file for comparison
# (Legacy file reading and comparison code will depend on your specific legacy file format)

# Stop MAPDL
pymapdl.mapdl('end')
```

This code does not include the geometry creation, mesh generation, boundary condition application, and legacy file reading parts, as those will depend on your specific pressure vessel and legacy file format. You'll need to fill in those parts according to your needs. Also, remember to install the pymapdl library before running the script. You can install it using pip:

```
pip install pymapdl
```