 from pyscf import gto, scf, cc

cell = gto.M(
    atom = [['C', (0, 0, 0)], ['C', (2.4987, 0, 0)]],
    basis = '6-31g',
    pseudopotential = 'gth-pade',
    a = [[5.4813, 0.0000, 0.0000], [0.0000, 5.4813, 0.0000], [0.0000, 0.0000, 5.4813]],
    unit = 'Bohr'
)

kpts = (2, 2, 2)

# KHF and KCCSD calculations with 2x2x2 k-points
mf_k = scf.KRHF(cell, kpts=kpts).run()
cc_k = cc.KCCSD(mf_k).run()
print("KCCSD energy per unit cell (2x2x2 k-points):", mf_k.e_tot + cc_k.e_corr)

mf_k = scf.KRHF(cell, kpts=(1, 1, 1)).run()
cc_k = cc.KCCSD(mf_k).run()
print("KCCSD energy per unit cell (1x1x1 k-point):", mf_k.e_tot + cc_k.e_corr)

# Single k-point calculation using RHF method
mf = scf.RHF(cell).run()
cc_r = cc.RCCSD(mf).run()
print("RCCSD energy per unit cell (single k-point):", mf.e_tot + cc_r.e_corr)

# RCCSD energy based on CCSD density matrices
e_rccsd = cc_r.kernel_energy()
print("RCCSD energy based on CCSD density matrices:", e_rccsd)

# UHF calculation
mf = scf.UHF(cell).run()
cc_u = cc.UCCSD(mf).run()
print("UCCSD energy per unit cell (single k-point):", mf.e_tot + cc_u.e_corr)

# UCCSD energy based on CCSD density matrices
e_uccsd = cc_u.kernel_energy()
print("UCCSD energy based on CCSD density matrices:", e_uccsd)

# GHF calculation
mf = scf.GHF(cell).run()
cc_g = cc.GCCSD(mf).run()
print("GCCSD energy per unit cell (single k-point):", mf.e_tot + cc_g.e_corr)

# GCCSD energy based on CCSD density matrices
e_gccsd = cc_g.kernel_energy()
print("GCCSD energy based on CCSD density matrices:", e_gccsd)