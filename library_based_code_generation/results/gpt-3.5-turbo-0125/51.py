import numpy as np
from pyscf import gto, scf, cc

# Build the cell
cell = gto.Cell()
cell.atom = [['C', (0, 0, 0)], ['C', (1.4, 1.4, 1.4)]]
cell.basis = 'sto-3g'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 3.5668
cell.unit = 'A'
cell.build()

# KHF and KCCSD calculations with 2x2x2 k-points
kpts = cell.make_kpts([2, 2, 2])
mf = scf.KRHF(cell, kpts).run()
mycc = cc.KRCCSD(mf).run()
print('Total energy per unit cell (2x2x2 k-points):', mycc.e_tot)

# KHF and KCCSD calculations for a single k-point
kpt = np.array([0, 0, 0])
mf = scf.KRHF(cell, kpt).run()
mycc = cc.KRCCSD(mf).run()
print('Total energy per unit cell (single k-point):', mycc.e_tot)

# RHF and RCCSD calculations for a single k-point
mf = scf.RHF(cell, kpt).run()
mycc = cc.RCCSD(mf).run()
print('Total energy per unit cell (RCCSD):', mycc.e_tot)
mycc = cc.RCCSD(mf)
mycc.kernel()
print('RCCSD energy based on CCSD density matrices:', mycc.e_corr)

# UHF and UCCSD calculations for a single k-point
mf = scf.UHF(cell, kpt).run()
mycc = cc.UCCSD(mf).run()
print('Total energy per unit cell (UCCSD):', mycc.e_tot)
mycc = cc.UCCSD(mf)
mycc.kernel()
print('UCCSD energy based on CCSD density matrices:', mycc.e_corr)

# GHF and GCCSD calculations for a single k-point
mf = scf.GHF(cell, kpt).run()
mycc = cc.GCCSD(mf).run()
print('Total energy per unit cell (GCCSD):', mycc.e_tot)
mycc = cc.GCCSD(mf)
mycc.kernel()
print('GCCSD energy based on CCSD density matrices:', mycc.e_corr)