from ansys.mapdl import launch_mapdl

mapdl = launch_mapdl()
mapdl.prep7()

mapdl.block(0, 1, 0, 1, 0, 1)
mapdl.et(1, "SOLID186")
mapdl.vmesh("ALL")

mapdl.block(0, 1, 0, 1, 1, 2)
mapdl.et(2, "SOLID187")
mapdl.vmesh("ALL")

mapdl.esel("SOLID", "TYPE", "", 1)
mapdl.esel("SOLID", "TYPE", "", 2)
mapdl.esel("SOLID", "INTERSECT")
mapdl.cm("SEL", "ADD", "CONTACT")
mapdl.cm("SEL", "OPERATE", "NODE")
mapdl.cm("SEL", "TYPE", "CONTACT")
mapdl.cm("SEL", "DELE", "ALL")
mapdl.cm("SEL", "DELE", "UNREF")
mapdl.cm("SEL", "DELE", "UNUSE")

mapdl.eplot("CONTACT")
mapdl.finish()