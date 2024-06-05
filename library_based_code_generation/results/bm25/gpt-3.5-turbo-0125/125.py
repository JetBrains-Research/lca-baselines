import numpy as np
from pyscf.pbc import gto, scf, mp

cell = gto.Cell()
cell.atom = '''H 0. 0. 0.; H 0. 0. 1.'''
cell.basis = 'sto-3g'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 4
cell.unit = 'Bohr'
cell.verbose = 3
cell.build()

kpts = cell.make_kpts([2, 2, 2])
mf = scf.KRHF(cell, kpts).run()
mp2 = mp.KMP2(mf).run()
print('KMP2 energy per unit cell:', mp2.e_tot)

kpts_single = np.array([0., 0., 0.])
mf_single = scf.KRHF(cell, kpts_single).run()
mp2_single = mp.KMP2(mf_single).run()
print('KMP2 energy per unit cell (single k-point):', mp2_single.e_tot)

rhf = scf.KRHF(cell, kpts_single).run()
rmp2 = mp.RMP2(rhf).run()
print('RMP2 energy per unit cell at the k-point:', rmp2.e_tot)

rdm1, rdm2 = rmp2.make_rdm1_and_rdm2()
total_energy = rmp2.energy(rdm1, rdm2)
print('Total energy based on MP2 density matrices:', total_energy)

uhf = rhf.to_uhf()
ghf = rhf.to_ghf()

ump2_uhf = mp.UMP2(uhf).run()
rdm1_uhf, rdm2_uhf = ump2_uhf.make_rdm1_and_rdm2()
total_energy_uhf = ump2_uhf.energy(rdm1_uhf, rdm2_uhf)
print('UMP2 energy per unit cell at the k-point:', total_energy_uhf)

gmp2_ghf = mp.GMP2(ghf).run()
rdm1_ghf, rdm2_ghf = gmp2_ghf.make_rdm1_and_rdm2()
total_energy_ghf = gmp2_ghf.energy(rdm1_ghf, rdm2_ghf)
print('GMP2 energy per unit cell at the k-point:', total_energy_ghf)