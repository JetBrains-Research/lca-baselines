```python
import numpy as np
import os
from pyscf import gto, dft, scf, ao2mo, lib

# Define the molecule
mol = gto.Mole()
mol.atom = '''
    H 0 0 0
    H 0 0 0.74
'''
mol.basis = 'cc-pVDZ'
mol.build()

# Perform DFT calculations for state 1
dft1 = dft.RKS(mol)
dft1.xc = 'B3LYP'
dft1.chkfile = 'state1.chk'
dft1.kernel()

# Perform DFT calculations for state 2
dft2 = dft.RKS(mol)
dft2.xc = 'B3LYP'
dft2.chkfile = 'state2.chk'
dft2.kernel()

# Read MO coefficients and occupation numbers from chkfiles
mo_coeff1 = scf.chkfile.load('state1.chk', 'scf/mo_coeff')
mo_occ1 = scf.chkfile.load('state1.chk', 'scf/mo_occ')
mo_coeff2 = scf.chkfile.load('state2.chk', 'scf/mo_coeff')
mo_occ2 = scf.chkfile.load('state2.chk', 'scf/mo_occ')

# Calculate overlap between two determinants
s = mol.intor_symmetric('int1e_ovlp')
overlap = np.dot(mo_coeff1.T, np.dot(s, mo_coeff2))

# Construct density matrices
dm1 = np.dot(mo_coeff1 * mo_occ1, mo_coeff1.T)
dm2 = np.dot(mo_coeff2 * mo_occ2, mo_coeff2.T)

# Calculate one-electron and two-electron part contributions
h1 = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
core_hamiltonian = np.einsum('ij,ji->', h1, dm1 + dm2)

eri = ao2mo.kernel(mol, mo_coeff1)
J1, K1 = scf.hf.get_jk(mol, dm1)
J2, K2 = scf.hf.get_jk(mol, dm2)
coulomb_repulsion = np.einsum('ij,ji->', J1 + J2, dm1 + dm2) / 2
exchange_repulsion = -np.einsum('ij,ji->', K1, dm1) / 2 - np.einsum('ij,ji->', K2, dm2) / 2

# Calculate new total energy
new_total_energy = core_hamiltonian + coulomb_repulsion + exchange_repulsion

# Calculate effective electronic coupling
effective_electronic_coupling = new_total_energy - (dft1.e_tot + dft2.e_tot) / 2

# Print the results
print("Effective Electronic Coupling:", effective_electronic_coupling)

# Remove chkfiles
os.remove('state1.chk')
os.remove('state2.chk')
```