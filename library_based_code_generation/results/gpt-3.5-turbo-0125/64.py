import os
from ansys.mapdl import launch_mapdl

mapdl = launch_mapdl()
mapdl.prep7()
mapdl.units("US")
mapdl.et(1, "SOLID186")
mapdl.mp("EX", 1, 30e6)
mapdl.mp("DENS", 1, 0.3)
mapdl.mp("NUXY", 1, 0.3)
mapdl.rectng(0, 10, 0, 10, 0, 10)
mapdl.vmesh("ALL")
mapdl.eplot()
mapdl.nsel("S", "LOC", "X", 0)
mapdl.d("ALL", "UX")
mapdl.nsel("S", "LOC", "X", 10)
mapdl.f("ALL", "FX", 100)
mapdl.solve()
mapdl.post1()
mapdl.set("contour", "vonmises")
mapdl.p1()
mapdl.finish()