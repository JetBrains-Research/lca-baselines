



 Here is the Python code that performs the task:

```
import numpy as np
from pyscf import gto, scf, dft, lib
from pyscf.tools import cubegen

# Define the molecule
mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 1.1'
mol.basis = 'cc-pvdz'
mol.build()

# Perform two state calculations with DFT
mf1 = dft.RKS(mol)
mf1.xc = 'b3lyp'
mf1.chkfile = 'mf1.chk'
mf1.kernel()

mf2 = dft.RKS(mol)
mf2.xc = 'b3lyp'
mf2.chkfile = 'mf2.chk'
mf2.kernel()

# Read the MO coefficients and occupation numbers from the chkfiles
mo_coeff1, mo_occ1 = lib.chkfile.load('mf1.chk', 'scf/mo_coeff'), lib.chkfile.load('mf1.chk', 'scf/mo_occ')
mo_coeff2, mo_occ2 = lib.chkfile.load('mf2.chk', 'scf/mo_coeff'), lib.chkfile.load('mf2.chk', 'scf/mo_occ')

# Calculate the overlap between two determinants
s12 = gto.intor_cross('int1e_ovlp', mol, mol)
s12 = reduce(np.dot, (mo_coeff1.T, s12, mo_coeff2))

# Construct density matrices
dm1 = np.dot(mo_coeff1[:,mo_occ1>0], mo_coeff1[:,mo_occ1>0].T)
dm2 = np.dot(mo_coeff2[:,mo_occ2>0], mo_coeff2[:,mo_occ2>0].T)

# Calculate one-electron and two-electron part contributions
e1 = np.einsum('ij,ji', dm1, mf1.get_hcore())
e2 = np.einsum('ij,ji', dm1, mf1.get_veff())

# Calculate new total energy
e_tot = mf1.energy_tot() + (e1 + e2) * s12

# Calculate the effective electronic coupling
ec = e_tot - mf1.energy_tot() - mf2.energy_tot()

# Print the results
print('Effective electronic coupling:', ec)

# Remove the chkfiles
os.remove('mf1.chk')
os.remove('mf2.chk')
```

This code defines a molecule with two hydrogen atoms, performs two state calculations with DFT, reads the MO coefficients and occupation numbers from the chkfiles, calculates the overlap between two determinants, constructs density matrices, calculates one-electron and two-electron part contributions, calculates the new total energy, calculates the effective electronic coupling, and prints the results. It also removes the chkfiles at the end.

Please note that this code is just a sample and may require modifications to fit your specific needs.

I hope this helps! 😊