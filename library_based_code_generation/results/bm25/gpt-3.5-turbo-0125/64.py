```python
from ansys.mapdl import launch_mapdl

mapdl = launch_mapdl()
mapdl.prep7()
mapdl.units('in')

mapdl.et(1, "SOLID186")
mapdl.mp("EX", 1, 30e6)
mapdl.mp("PRXY", 1, 0.3)

mapdl.block(0, 10, 0, 10, 0, 10)
mapdl.vmesh("ALL")

mapdl.nsel("S", "LOC", "X", 0)
mapdl.d("ALL", "UX")
mapdl.nsel("S", "LOC", "X", 10)
mapdl.f("ALL", "FX", 100)

mapdl.nsel("S", "LOC", "Z", 0)
mapdl.d("ALL", "UZ")
mapdl.nsel("S", "LOC", "Z", 10)
mapdl.f("ALL", "FZ", 200)

mapdl.solve()
mapdl.finish()

mapdl.post1()
mapdl.set("esel", "s", "mat", 1)
mapdl.post_processing.plot_von_mises()
mapdl.finish()
```