  from pyscf import gto, scf, dft
import numpy as np

# Define molecule with specific atoms and basis
mol = gto.M(
    atom='H 0.75 0.75 0.75; H 0.75 0.75 1.25',
    basis='6-31g'
)

# Perform two state calculations with DFT, storing MO information into separate chkfiles
scf.RHF(mol).run()
scf.RHF(mol).run(chkfile='state1.chk')
scf.RHF(mol).run(chkfile='state2.chk')

# Read MO coefficients and occupation numbers from chkfiles
mo_coeff1 = np.loadtxt('state1.chk', skiprows=2)
mo_coeff2 = np.loadtxt('state2.chk', skiprows=2)
occ1 = np.loadtxt('state1.chk', skiprows=2, usecols=(3,))
occ2 = np.loadtxt('state2.chk', skiprows=2, usecols=(3,))

# Calculate overlap between two determinants
overlap = np.dot(mo_coeff1.T, mo_coeff2)

# Construct density matrices
density1 = np.dot(mo_coeff1, np.dot(occ1, mo_coeff1.T))
density2 = np.dot(mo_coeff2, np.dot(occ2, mo_coeff2.T))

# Calculate one-electron and two-electron part contributions
one_electron = np.einsum('ij,ji->', density1, density2)
two_electron = np.einsum('ijkl,ijkl->', density1, density2)

# Calculate new total energy
new_total_energy = one_electron + two_electron

# Calculate effective electronic coupling
eff_ecoupling = np.abs(new_total_energy - scf.RHF(mol).energy_tot())

# Print results
print('Effective electronic coupling:', eff_ecoupling)

# Remove chkfiles
import os
os.remove('state1.chk')
os.remove('state2.chk')