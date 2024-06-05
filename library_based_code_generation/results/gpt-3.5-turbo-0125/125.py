cell = pyscf.pbc.gto.Cell()
cell.atom = '''H 0. 0. 0.; H 0. 0. 1.'''
cell.basis = 'sto-3g'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 4
cell.unit = 'Bohr'
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 2, 2])
mf = pyscf.pbc.scf.KRHF(cell, kpts).run()
mp2 = pyscf.pbc.mp.MP2(mf).run()

print('KMP2 energy per unit cell:', mp2.e_tot)

kpts_single = np.array([0, 0, 0])
mf_single = pyscf.pbc.scf.KRHF(cell, kpts_single).run()
mp2_single = pyscf.pbc.mp.MP2(mf_single).run()

print('KMP2 energy per unit cell (single k-point):', mp2_single.e_tot)

rhf = pyscf.scf.RHF(cell).run()
rmp2 = pyscf.mp.MP2(rhf).run()

print('RMP2 energy per unit cell at the k-point:', rmp2.e_tot)

rdm1, rdm2 = rmp2.make_rdm1(), rmp2.make_rdm2()
total_energy = rmp2.energy_tot(rdm1, rdm2)

uhf = rhf.to_uhf()
ghf = rhf.to_ghf()

ump2_uhf = pyscf.mp.UMP2(uhf).run()
ump2_ghf = pyscf.mp.GMP2(ghf).run()

rdm1_uhf, rdm2_uhf = ump2_uhf.make_rdm1(), ump2_uhf.make_rdm2()
total_energy_uhf = ump2_uhf.energy_tot(rdm1_uhf, rdm2_uhf)

rdm1_ghf, rdm2_ghf = ump2_ghf.make_rdm1(), ump2_ghf.make_rdm2()
total_energy_ghf = ump2_ghf.energy_tot(rdm1_ghf, rdm2_ghf)

print('UMP2 energy per unit cell at the k-point:', total_energy_uhf)
print('GMP2 energy per unit cell at the k-point:', total_energy_ghf)
print('Total energy based on MP2 density matrices:', total_energy)