  from pyscf import gto, scf, mcscf, wannier90
from pywannier90 import Wannier90
import numpy as np

# Define unit cell
cell = gto.Cell()
cell.atom = 'C 0.0 0.0 0.0; C 0.75 0.75 0.75'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.build()

# Perform PBE calculation
kks = scf.KKS(cell)
kks.kernel()

# Save and load kks object
kks.save('kks.h5')
kks = scf.KKS(cell)
kks.load('kks.h5')

# Construct MLWFs
mlwfs = Wannier90(kks, num_wann=20, num_bands=kks.get_nbands(), num_kpts=kks.get_nkpts())
mlwfs.run()

# Export MLWFs in xsf format for plotting
mlwfs.export_xsf('mlwfs.xsf')

# Export certain matrices
kks.export_matrix('hamiltonian', 'hamiltonian.npz')
kks.export_matrix('overlap', 'overlap.npz')

# Run wannier90 using MLWFs
w90 = Wannier90(kks, num_wann=20, num_bands=kks.get_nbands(), num_kpts=kks.get_nkpts())
w90.run()

# Interpolate Fock or band structure using Slater-Koster scheme
scf.get_bands(kks, kpts=kks.get_kpts(), nbands=kks.get_nbands(), with_fake=True)

# Print difference in eigenvalues interpolated by scf.get_bands function and by pywannier90
print(np.abs(kks.get_bands(kpts=kks.get_kpts(), nbands=kks.get_nbands(), with_fake=True) - w90.get_bands()))

# Plot band structure using mcu
mcu = w90.get_mcu()
mcu.plot_band_structure(kpts=kks.get_kpts(), nbands=kks.get_nbands(), with_fake=True)