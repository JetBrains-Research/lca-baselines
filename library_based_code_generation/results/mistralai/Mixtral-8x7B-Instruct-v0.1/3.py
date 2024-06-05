from pyscf import gto, scf, dft
from pyscf.tools import molden
import numpy as np

# Define molecule
mol = gto.M(
    atom='H 0 0 0; F 0 0 1.1',
    basis='6-31g',
)

# Perform two state calculations with DFT
mf1 = dft.RKS(mol)
mf1.xc = 'blyp'
mf1.kernel()
mf1.chkfile = 'chkfile1.chk'
molden.from_scf(mf1, 'det1.molden')

mf2 = dft.RKS(mol)
mf2.xc = 'blyp'
mf2.kernel()
mf2.chkfile = 'chkfile2.chk'
molden.from_scf(mf2, 'det2.molden')

# Read MO coefficients and occupation numbers
mo1 = np.load('{}.mo_coeff'.format(mf1.chkfile))
occ1 = mf1.scf._finalize()['mo_occ']
mo2 = np.load('{}.mo_coeff'.format(mf2.chkfile))
occ2 = mf2.scf._finalize()['mo_occ']

# Calculate overlap between two determinants
s = np.dot(mo1.T, mo2)

# Construct density matrices
dm1 = np.diag(occ1)
dm2 = np.diag(occ2)

# Calculate one-electron and two-electron part contributions
h1e = mol.intor('int1e_nuc_sph', comp=3)
h1e_part = np.einsum('ij,ji->', h1e, s) + np.einsum('ij,ji->', s, h1e)

nav = mol.nao
eris = scf.eris_df(mf1, mf2)
v2e_part = np.einsum('ijab,jiab->', eris.ooov[0], s) + np.einsum('ijab,ijab->', eris.ovoo[0], s)

# Calculate new total energy
new_energy = h1e_part + 0.5 * v2e_part

# Calculate the effective electronic coupling
eff_coupling = new_energy[0, 1]

print('Effective electronic coupling:', eff_coupling)

# Remove chkfiles
import os
os.remove('chkfile1.chk')
os.remove('chkfile2.chk')