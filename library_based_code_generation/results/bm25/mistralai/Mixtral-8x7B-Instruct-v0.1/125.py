 from pyscf.pbc import gto, scf, mp

cell = gto.Cell()
cell.atom = [[0, 0, 0], [1.63, 0, 0], [0, 1.63, 0], [1.63, 1.63, 0]]
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 1.63
cell.unit = 'B'
cell.verboose = 4
cell.build()

kpts = cell.make_kpts([2, 2, 2])

khf_scf = scf.KHF(cell, kpts).run()
khf_mp2 = mp.KMP2(khf_scf).run()
print("KMP2 energy per unit cell with 2x2x2 k-points:", khf_mp2.e_mp2)

khf_scf_k = scf.KHF(cell, kpts[0]).run()
khf_mp2_k = mp.KMP2(khf_scf_k).run()
print("KMP2 energy per unit cell with 1 k-point:", khf_mp2_k.e_mp2)

rhf_scf_k = scf.RHF(cell, kpts[0]).run()
rhf_mp2_k = mp.RMP2(rhf_scf_k).run()
print("RMP2 energy per unit cell with 1 k-point:", rhf_mp2_k.e_mp2)

dms = rhf_mp2_k.make_rdm12()
print("Total energy based on MP2 density matrices:", rhf_scf_k.energy_total(dms))

uhf_scf_k = scf.UHF(cell, kpts[0]).run()
uhf_mp2_k = mp.UMP2(uhf_scf_k).run()
print("UMP2 energy per unit cell with 1 k-point:", uhf_mp2_k.e_mp2)

ghf_scf_k = scf.GHF(cell, kpts[0]).run()
ghf_mp2_k = mp.GMP2(ghf_scf_k).run()
print("GMP2 energy per unit cell with 1 k-point:", ghf_mp2_k.e_mp2)

dms_uhf = uhf_mp2_k.make_rdm12()
print("Total energy based on UMP2 density matrices:", uhf_scf_k.energy_total(dms_uhf))

dms_ghf = ghf_mp2_k.make_rdm12()
print("Total energy based on GMP2 density matrices:", ghf_scf_k.energy_total(dms_ghf))