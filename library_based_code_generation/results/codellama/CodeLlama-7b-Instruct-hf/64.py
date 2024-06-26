  from pymapdl import *

# Launch MAPDL and set units to US Customary system
mapdl = launch_mapdl()
mapdl.prep7()
mapdl.units("IN")

# Define materials and element type
mapdl.mp("EX", 1, 7.85e-10, 0.3) # Steel
mapdl.emodif("ALL", "MAT", 1)
mapdl.et(1, "SOLID180")

# Create geometry of pressure vessel
mapdl.block(0, 0, 0, 1, 1, 1)
mapdl.vmesh("ALL")

# Create mesh
mapdl.mesh("ALL")

# Apply boundary conditions and pressure
mapdl.nsel("S", "LOC", "Z", 0)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 1)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 2)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 3)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 4)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 5)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 6)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 7)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 8)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 9)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 10)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 11)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 12)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 13)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 14)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 15)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 16)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 17)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 18)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 19)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 20)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 21)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 22)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 23)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 24)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 25)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 26)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 27)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 28)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 29)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 30)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 31)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 32)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 33)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 34)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 35)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 36)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 37)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 38)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 39)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 40)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 41)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 42)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 43)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 44)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 45)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 46)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 47)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 48)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 49)
mapdl.d("ALL", "ALL")
mapdl.allsel()
mapdl.nsel("S", "LOC", "Z", 50)
mapdl.d("