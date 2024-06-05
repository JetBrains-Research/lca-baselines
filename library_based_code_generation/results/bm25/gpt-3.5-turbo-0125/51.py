import numpy as np
from pyscf import gto, scf, cc

# Build the cell
cell = gto.Cell()
cell.atom = [['C', (0, 0, 0)], ['C', (1.4, 0, 0)]]
cell.basis = 'sto-3g'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 3.4
cell.unit = 'A'
cell.build()

# KHF and KCCSD calculations with 2x2x2 k-points
kpts = cell.make_kpts([2, 2, 2])
mf = scf.KRHF(cell, kpts).run()
ccsd = cc.KRCCSD(mf).run()
print(ccsd.e_tot / cell.nao)

# KHF and KCCSD calculations for a single k-point
kpt = np.array([0, 0, 0])
mf = scf.KRHF(cell, kpt).run()
ccsd = cc.KRCCSD(mf).run()
print(ccsd.e_tot / cell.nao)

# RHF and RCCSD calculations
mf = scf.RHF(cell, kpt).run()
ccsd = cc.RCCSD(mf).run()
print(ccsd.e_tot / cell.nao)
ccsd = cc.RCCSD(mf)
ccsd.kernel()
print(ccsd.e_corr)

# UHF and UCCSD calculations
mf = scf.UHF(cell, kpt).run()
uccsd = cc.UCCSD(mf).run()
print(uccsd.e_tot / cell.nao)
uccsd = cc.UCCSD(mf)
uccsd.kernel()
print(uccsd.e_corr)

# GHF and GCCSD calculations
mf = scf.GHF(cell, kpt).run()
gccsd = cc.GCCSD(mf).run()
print(gccsd.e_tot / cell.nao)
gccsd = cc.GCCSD(mf)
gccsd.kernel()
print(gccsd.e_corr)