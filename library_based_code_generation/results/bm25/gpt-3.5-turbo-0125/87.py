```python
from ansys.mapdl import core as pymapdl

mapdl = pymapdl.launch_mapdl()

mapdl.prep7()
mapdl.block(0, 1, 0, 1, 0, 1)
mapdl.et(1, "SOLID186")
mapdl.vmesh("ALL")
mapdl.block(0, 1, 0, 1, 1, 2)
mapdl.et(2, "SOLID187")
mapdl.vmesh("ALL")
mapdl.esel("S", "TYPE", "", 1)
mapdl.esel("A", "TYPE", "", 2)
mapdl.contact("GENERIC")
mapdl.finish()
mapdl.finish()
mapdl.post1()
mapdl.plo("ELEM")
mapdl.exit()
```