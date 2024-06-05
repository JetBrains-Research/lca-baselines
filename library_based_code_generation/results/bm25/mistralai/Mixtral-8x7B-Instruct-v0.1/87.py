from pymapdl import Model

with Model() as mapdl:
    mapdl.prep7()

    # Create first block and mesh it with tetrahedral elements
    mapdl.block(size=(10, 10, 10))
    mapdl.mesh(size=(10, 10, 1