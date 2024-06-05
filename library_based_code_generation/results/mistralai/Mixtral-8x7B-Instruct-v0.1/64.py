 ```python
import pymapdl

with pymapdl.open_mapdl(debug=True) as mapdl:
    mapdl.units("inch")

    # Define materials and element type
    mapdl.material("mat1", 30000, 0.3)
    mapdl.et("ELFORM", 1, "S4R", "RARE", is_add=True)

    # Create geometry of the pressure vessel
    mapdl.blc4(0, 0, 0, 0)
    mapdl.blc4(10, 0, 0, 0)
    mapdl.blc4(10, 10, 0, 0)
    mapdl.blc4(0, 10, 0, 0)
    mapdl.bsol(10, 10, 1, 0, 0)
    mapdl.csys(4, 0)
    mapdl.cgen(0, 0, 0, 1, 1, 0, 0, 0)
    mapdl.csys(0, 0)

    # Create mesh
    mapdl.esize(1.0)
    mapdl.esm(1)

    # Apply boundary conditions and pressure
    mapdl.nsel("S", "LOC", "Z", 0)
    mapdl.d("ALL", "UX", "UX", 0)
    mapdl.d("ALL", "UY", "UY", 0)
    mapdl.d("ALL", "UR", "UR", 0)
    mapdl.sfe("ALL", 1, 1500)

    # Solve the problem
    mapdl.ants()
    mapdl.solve()

    # Post-process results
    mapdl.nsel("S", "LOC", "Z", 10)
    mapdl.get("VMIS", "MAX", "MAX")
    print(f"Von-Mises stress for the single static solution: {mapdl.valmax}")

    # Plot the results
    mapdl.plot(style="FEQ")

    # Compare results with the legacy file reader
    # (Assuming the legacy file "legacy.anc" is available)
    mapdl.legacy_read("legacy.anc")
    mapdl.compare_results()

    # Stop MAPDL
    mapdl.exit()
```