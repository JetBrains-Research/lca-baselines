 Here is a Python script that uses the pymapdl library to create the desired contact elements:

```python
from pymapdl.mapdl import MAPDL, Preprocessor, Elements, BirthAndDeath, cube_geom_and_mesh, a, test_elements, test_non_existing_elements, test_unselected_elements, _load_elements_offset, create, nodal_contact_friction_stress, plot_nodal_contact_friction_stress, get_elements_per_area

# Launch MAPDL and enter pre-processor
mapdl = MAPDL()
mapdl.run('PREP7')

# Create a block and mesh it with tetrahedral elements
block_name = 'Block1'
cube_geom_and_mesh(block_name, 0, 0, 0, 1, 1, 1, 'TET4')

# Create a second volume block above the existing one and mesh it with quadratic hexahedral elements
block_name_2 = 'Block2'
cube_geom_and_mesh(block_name_2, 0, 0, 1, 1, 1, 1, 'C3D8R')

# Select all the elements at the intersection between the two blocks
elements_at_intersection = set()
for elements in Elements():
    if test_elements(elements, 'TET4') and test_elements(_load_elements_offset(elements), 'C3D8R'):
        elements_at_intersection.add(elements)

# Create contact elements
contact_elements = set()
for element in elements_at_intersection:
    contact_elements.add(create('CONTACT', element))

# Plot the contact element pairs as a wire-frame
plot_nodal_contact_friction_stress(wireframe=True)

# Solve contact problem
BirthAndDeath('CONTACT')
contact_solve()

# Stop MAPDL
mapdl.run('END')
```

Please note that you need to have the pymapdl library installed to run this script. If it's not installed, you can install it using pip:

```
pip install pymapdl
```

Also, this script assumes that the blocks are positioned correctly so that they do not touch. You may need to adjust the block positions or the mesh sizes to ensure that they do not overlap.