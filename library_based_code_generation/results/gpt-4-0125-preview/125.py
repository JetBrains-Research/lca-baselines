```python
from pyscf import pbc, gto, scf, mp
import numpy as np

# Define cell
cell = pbc.gto.Cell()
cell.atom = 'He 0 0 0; He 1 0 0; He 0 1 0; He 0 0 1'
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 4.0
cell.unit = 'B'
cell.verbose = 5
cell.build()

# KHF and KMP2 calculations with 2x2x2 k-points
kpts = cell.make_kpts([2, 2, 2])
khf = scf.KRHF(cell, kpts=kpts).run()
kmp2 = mp.KMP2(khf).run()
print("KMP2 energy per unit cell (2x2x2 k-points):", kmp2.e_tot)

# KHF and KMP2 calculations with a single k-point
kpts = cell.make_kpts([1, 1, 1])
khf = scf.KRHF(cell, kpts=kpts).run()
kmp2 = mp.KMP2(khf).run()
print("KMP2 energy per unit cell (single k-point):", kmp2.e_tot)

# Single k-point calculation using RHF method
rhf = scf.RHF(cell).run()
rmp2 = mp.RMP2(rhf).run()
print("RMP2 energy per unit cell at the k-point:", rmp2.e_tot)

# First and second order reduced density matrices
dm1 = rmp2.make_rdm1()
dm2 = rmp2.make_rdm2()
energy_from_dm = np.einsum('ij,ji->', dm1, rhf.get_hcore()) + \
                 np.einsum('ijkl,ijkl->', dm2, rhf._eri) / 2
print("Total energy from density matrices:", energy_from_dm)

# Convert RHF to UHF and GHF, then perform UMP2 and GMP2 calculations
uhf = rhf.to_uhf()
ghf = rhf.to_ghf()

ump2 = mp.UMP2(uhf).run()
gmp2 = mp.GMP2(ghf).run()

# UMP2 and GMP2 energy per unit cell at the k-point
print("UMP2 energy per unit cell at the k-point:", ump2.e_tot)
print("GMP2 energy per unit cell at the k-point:", gmp2.e_tot)

# Calculate total energy based on the MP2 density matrices for UHF and GHF
dm1_u = ump2.make_rdm1()
dm2_u = ump2.make_rdm2()
energy_from_dm_u = np.einsum('ij,ji->', dm1_u, uhf.get_hcore()) + \
                   np.einsum('ijkl,ijkl->', dm2_u, uhf._eri) / 2

dm1_g = gmp2.make_rdm1()
dm2_g = gmp2.make_rdm2()
energy_from_dm_g = np.einsum('ij,ji->', dm1_g, ghf.get_hcore()) + \
                   np.einsum('ijkl,ijkl->', dm2_g, ghf._eri) / 2

print("Total energy from UMP2 density matrices:", energy_from_dm_u)
print("Total energy from GMP2 density matrices:", energy_from, "_dm_g")
```