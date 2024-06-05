 import pyscf
import os

mol = pyscf.gto.Mole()
mol.atom = [['O', (0, 0, 0)], ['H', (0, 0, 1.1)]]
mol.basis = '6-31g'
mol.build()

mf1 = pyscf.scf.RHF(mol)
mf1.xc = format_xc_code('b3lyp')
mf1.kernel()
mf1.chkfile = 'diabatic1.chk'
mf1.save(mf1.chkfile)

mf2 = pyscf.scf.RHF(mol)
mf2.xc = format_xc_code('b3lyp')
mf2.kernel()
mf2.chkfile = 'diabatic2.chk'
mf2.save(mf2.chkfile)

mo1 = _load_and_unpack(mf1.chkfile, mol)
mo2 = _load_and_unpack(mf2.chkfile, mol)

occ1 = mo1[1]
occ2 = mo2[1]

coeff1 = mo1[0]
coeff2 = mo2[0]

S = pyscf.ao2mo.restore(1, calculate(mol, coeff1, coeff2), mol.nao_nr())

D1 = np.einsum('ij,ji->', coeff1, occ1)
D2 = np.einsum('ij,ji->', coeff2, occ2)

h1e = calculate_energy(mol, coeff1, hermi=1)
h2e = calculate_energy(mol, coeff2, hermi=1)

eris = calculate_integrals_(mol, coeff1, coeff2, hermi=2)
v2e = eris[0] + eris[1]

Hij = h1e + h2e + np.einsum('ijab,ij->ab', S, v2e)

e1, c1 = two_pole(Hij, S)

os.remove('diabatic1.chk')
os.remove('diabatic2.chk')

print('Effective electronic coupling:', e1)