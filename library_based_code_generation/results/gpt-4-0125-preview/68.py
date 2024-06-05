```python
from pyscf import gto, scf, agf2, lib
from pyscf.agf2 import energy

# Define the molecule
mol = gto.Mole()
mol.atom = '''
    O 0.0 0.0 0.0
    H 0.0 -0.757 0.587
    H 0.0 0.757 0.587
'''
mol.basis = 'cc-pVDZ'
mol.build()

# Run RHF calculation with specified convergence tolerance
mf = scf.RHF(mol)
mf.conv_tol = 1e-8
mf.kernel()

# Run AGF2 calculation with specified convergence tolerance
auxbasis = 'cc-pVDZ-ri'
gf2 = agf2.AGF2(mf, auxbasis=auxbasis)
gf2.conv_tol = 1e-6
gf2.kernel()

# Print the first three ionization potentials and electron affinities
print("Ionization potentials:", gf2.ipagf2(nroots=3)[0])
print("Electron affinities:", gf2.eaagf2(nroots=3)[0])

# Calculate the MO-basis density matrix
dm_mo = gf2.make_rdm1()

# Calculate dipole moments
dip_ints = mol.intor_symmetric('int1e_r', comp=3)
dm_ao = scf.addons.mo2ao(dm_mo, mf.mo_coeff)
dip_mom = lib.einsum('xij,ji->x', dip_ints, dm_ao)

# Add the nuclear component to the dipole moment
dip_mom += mol.atom_charges() @ mol.atom_coords()

# Print the calculated dipole moment
print("Dipole moment (Debye):", dip_mom * lib.param.DEBYE)
```