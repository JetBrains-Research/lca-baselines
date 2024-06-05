 Here is a Python script that uses the pymapdl library to create the desired contact elements:

```python
from pymapdl import mapdl, pre, post, mesh, analysis, contact

# Launch MAPDL and enter pre-processor
mapdl.run('mapdl')
pre.enter()

# Create a block and mesh it with tetrahedral elements
block_name = 'Block1'
pre.block(name=block_name, x1=0, y1=0, z1=0, x2=1, y2=1, z2=1, element='C3D10')
pre.mesh(block_name)

# Create a second volume block above the existing one and mesh it with quadratic hexahedral elements
block_name_2 = 'Block2'
pre.block(name=block_name_2, x1=0, y1=0, z1=1.1, x2=1, y2=1, z2=2.1, element='C3D20')
pre.mesh(block_name_2)

# Ensure that the blocks do not touch
pre.offset(block_name_2, direction='Z', distance=0.1)

# Select all the elements at the intersection between the two blocks and generate contact elements
contact_pair_name = 'ContactPair'
pre.select('ELEMENT', block_name + ' INTERSECT ' + block_name_2)
contact.create(contact_pair_name, 'SURFACE-TO-SURFACE')

# Plot the contact element pairs as a wire-frame
post.enter()
post.plot(contact_pair_name, style='WIREFRAME')

# Stop MAPDL
mapdl.quit()
```

Please note that you need to have the pymapdl library installed in your Python environment to run this script. If you haven't installed it yet, you can do so using pip:

```bash
pip install pymapdl
```