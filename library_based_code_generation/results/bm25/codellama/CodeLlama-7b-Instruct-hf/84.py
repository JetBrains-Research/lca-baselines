  Here is the Python code that constructs Maximally Localized Wannier Functions (MLWFs) using the pywannier90 tool from the pyscf library:
```
from pyscf import gto, scf, mcscf, wannier90

# Define a unit cell
cell = gto.Cell()
cell.atom = 'C 0.0 0.0 0.0; C 0.75 0.75 0.75'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.build()

# Perform a PBE calculation
kpts = cell.make_kpts([2, 2, 2])
kmesh = [2, 2, 2]
scf.RHF(cell, kpts).kernel()

# Save and load the kks object
kks = scf.KKS(cell, kpts)
kks.kernel()

# Construct MLWFs
mlwfs = wannier90.MLWFs(kks, kpts, kmesh)

# Export the MLWFs in xsf format for plotting
mlwfs.export_xsf('mlwfs.xsf')

# Export certain matrices
matrices = [mlwfs.A, mlwfs.B, mlwfs.C]

# Run a wannier90 using these
wannier90.run(kks, kpts, kmesh, matrices)

# Interpolate the Fock or band structure using the Slater-Koster scheme
fock = scf.get_bands(kks, kpts, kmesh)
bands = wannier90.interpolate_band(kks, kpts, kmesh, fock)

# Print the difference in the eigenvalues interpolated by scf.get_bands function and by pywannier90
print(bands - fock)

# Plot the band structure using mcu
mcu = wannier90.MCU(kks, kpts, kmesh)
mcu.plot_band(bands)
```
Note that this code assumes that you have already installed the pyscf library and the pywannier90 tool. Additionally, you will need to have the xsf format installed in order to export the MLWFs in xsf format.