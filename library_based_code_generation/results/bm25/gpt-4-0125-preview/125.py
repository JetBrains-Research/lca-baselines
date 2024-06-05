```python
from pyscf import pbc, gto, scf, mp
import numpy as np

# Create a cell
cell = pbc.gto.Cell()
cell.atom = 'He 0 0 0; He 1 0 0'
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 4.0
cell.unit = 'B'
cell.verbose = 5
cell.build()

# K-point calculations
kpts = cell.make_kpts([2, 2, 2])

# KHF calculation
khf = scf.KRHF(cell, kpts=kpts).run()

# KMP2 calculation
kmp2 = mp.KMP2(khf).run()
print("KMP2 energy per unit cell (2x2x2 k-points):", kmp2.e_tot)

# Single k-point calculations
kpts = cell.make_kpts([1, 1, 1])

# KHF calculation
khf = scf.KRHF(cell, kpts=kpts).run()

# KMP2 calculation
kmp2 = mp.KMP2(khf).run()
print("KMP2 energy per unit cell (single k-point):", kmp2.e_tot)

# RHF calculation
rhf = scf.RHF(cell).run()

# RMP2 calculation
rmp2 = mp.RMP2(rhf).run()
print("RMP2 energy per unit cell at the k-point:", rmp2.e_tot)

# First and second order reduced density matrices
dm1 = rmp2.make_rdm1()
dm2 = rmp2.make_rdm2()

# Total energy based on these matrices
e_tot = rmp2.energy(dm1, dm2)
print("Total energy based on the MP2 density matrices:", e_tot)

# Convert RHF to UHF and GHF
uhf = rhf.convert_to_uhf()
ghf = rhf.convert_to_ghf()

# UMP2 calculation
ump2 = mp.UMP2(uhf).run()
print("UMP2 energy per unit cell at the k-point:", ump2.e_tot)

# GMP2 calculation
gmp2 = mp.GMP2(ghf).run()
print("GMP2 energy per unit cell at the k-point:", gmp2.e_tot)

# UHF and GHF density matrices
dm1_uhf = ump2.make_rdm1()
dm2_uhf = ump2.make_rdm2()
dm1_ghf = gmp2.make_rdm1()
dm2_ghf = gmp2.make_rdm2()

# Total energy based on the UMP2 and GMP2 density matrices
e_tot_uhf = ump2.energy(dm1_uhf, dm2_uhf)
e_tot_ghf = gmp2.energy(dm1_ghf, dm2_ghf)

print("Total energy based on the UMP2 density matrices:", e_tot_uhf)
print("Total energy based on the GMP2 density matrices:", e_tot_ghf)
```