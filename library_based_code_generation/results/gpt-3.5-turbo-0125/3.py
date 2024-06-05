import numpy as np
from pyscf import gto, dft, lib

# Define molecule
mol = gto.Mole()
mol.atom = '''
C 0.0000 0.0000 0.0000
H 0.7572 0.5863 0.0000
H -0.7572 0.5863 0.0000
'''
mol.basis = 'sto-3g'
mol.build()

# Perform DFT calculations for two states
mf1 = dft.RKS(mol)
mf1.chkfile = 'state1.chk'
mf1.kernel()

mf2 = dft.RKS(mol)
mf2.chkfile = 'state2.chk'
mf2.kernel()

# Read MO coefficients and occupation numbers
mo_coeff1 = lib.chkfile.load('state1.chk', 'scf/mo_coeff')
mo_occ1 = lib.chkfile.load('state1.chk', 'scf/mo_occ')
mo_coeff2 = lib.chkfile.load('state2.chk', 'scf/mo_coeff')
mo_occ2 = lib.chkfile.load('state2.chk', 'scf/mo_occ')

# Calculate overlap between determinants
S = np.dot(mo_coeff1.T, mo_coeff2)

# Construct density matrices
P1 = np.dot(mo_coeff1 * mo_occ1, mo_coeff1.T)
P2 = np.dot(mo_coeff2 * mo_occ2, mo_coeff2.T)

# Calculate one-electron and two-electron part contributions
H = mf1.get_hcore()
J, K = mf1.get_jk()
F = H + 2*J - K
H1e = np.trace(np.dot((F + mf1.get_veff(mol, P1)), S))

# Calculate new total energy
E_total = mf1.energy_tot() + H1e

# Calculate effective electronic coupling
J = 2 * np.trace(np.dot(F, S))
K = np.trace(np.dot(np.dot(F, S), F))
V = J - K / 2

print("Effective electronic coupling:", V)

# Remove chkfiles
import os
os.remove('state1.chk')
os.remove('state2.chk')