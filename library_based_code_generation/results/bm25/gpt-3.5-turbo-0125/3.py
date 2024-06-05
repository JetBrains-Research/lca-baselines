import pyscf

# Define molecule
mol = pyscf.gto.Mole()
mol.atom = '''
O 0.000000 0.000000 0.000000
H 0.757459 0.586254 0.000000
H -0.757459 0.586254 0.000000
'''
mol.basis = 'sto-3g'
mol.build()

# Perform DFT calculations for two states
mf1 = pyscf.dft.RKS(mol)
mf1.xc = 'b3lyp'
mf1.chkfile = 'state1.chk'
mf1.kernel()

mf2 = pyscf.dft.RKS(mol)
mf2.xc = 'b3lyp'
mf2.chkfile = 'state2.chk'
mf2.kernel()

# Read MO coefficients and occupation numbers
mo_coeff1, mo_occ1 = pyscf.tools.chkfile.load('state1.chk', 'scf/mo_coeff', 'scf/mo_occ')
mo_coeff2, mo_occ2 = pyscf.tools.chkfile.load('state2.chk', 'scf/mo_coeff', 'scf/mo_occ')

# Calculate overlap between determinants
S = pyscf.tools.mo_overlap(mol, mo_coeff1, mo_coeff2)

# Construct density matrices
dm1 = pyscf.dft.numint.make_rdm1(mo_coeff1, mo_occ1)
dm2 = pyscf.dft.numint.make_rdm1(mo_coeff2, mo_occ2)

# Calculate one-electron and two-electron part contributions
h1e = pyscf.scf.hf.get_hcore(mol)
v2e = pyscf.scf.hf.get_veff(mol, dm1+dm2)

# Calculate new total energy
E_total = pyscf.scf.hf.energy_elec(mf1, dm1) + pyscf.scf.hf.energy_elec(mf2, dm2) + pyscf.scf.hf.energy_nuc(mol)

# Calculate effective electronic coupling
J = pyscf.lib.direct_sum('i,j->ij', mo_occ1, mo_occ2) * (S.T @ (h1e + v2e) @ S)

print("Effective electronic coupling:", J)

# Remove chkfiles
import os
os.remove('state1.chk')
os.remove('state2.chk')