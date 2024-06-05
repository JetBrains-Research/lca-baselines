import numpy as np
from pyscf import gto, scf, agf2, df

mol = gto.Mole()
mol.atom = '''
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
'''
mol.basis = 'cc-pvdz'
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-8
mf.kernel()

auxbasis = 'cc-pvdz-jkfit'
agf2_conv_tol = 1e-6

agf2_calc = agf2.AGF2(mf, auxbasis=auxbasis)
agf2_calc.conv_tol = agf2_conv_tol
agf2_calc.kernel()

ip_ea = agf2_calc.ip_ea(nroots=3)
print("Ionization Potentials and Electron Affinities:")
print(ip_ea)

dm_mo = mf.make_rdm1()
dip_ints = mol.intor('int1e_r')
dip_mo = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dip_ints, mf.mo_coeff)
dip_nuc = mol.atom_charges().sum(0) * mol.atom_coords().mean(0)
dip_total = dip_mo + dip_nuc
print("Dipole Moment (Debye):")
print(np.linalg.norm(dip_total))