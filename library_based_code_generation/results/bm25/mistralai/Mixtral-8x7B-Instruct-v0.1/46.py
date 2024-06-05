 ```python
from pyscf import gto, scf, fci, symm
import numpy as np

# Task 1: Orbital transformation of FCI wavefunction
mol1 = gto.Mole()
mol1.atom = 'H 0 0 0; H 0 0 1.2'
mol1.basis = 'sto-3g'
mol1.build()

mol2 = gto.Mole()
mol2.atom = 'H 0 0 0; He 0 0 1.2'
mol2.basis = 'sto-3g'
mol2.build()

mf1 = scf.RHF(mol1).run()
mf2 = scf.RHF(mol2).run()

fcisolver1 = fci.FCISolver(mol1)
fcisolver2 = fci.FCISolver(mol2)

wf1 = fcisolver1.kernel(mf1.mo_coeff)
wf2 = fcisolver2.kernel(mf2.mo_coeff)

C12 = symm.inter_mo_transform(mol1, mol2, mf1.mo_coeff, mf2.mo_coeff)
wf1_transformed = np.dot(wf1, np.dot(C12, C12.T))

assert symm.test_from_to_fci(mol1, mol2, wf1_transformed, wf2) < 1e-10

# Task 2: Transfer FCI wavefunction to a larger orbital space
mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 1.2'
mol.basis = 'sto-3g'
mol.build()

mf = scf.RHF(mol).run()

fcisolver = fci.FCISolver(mol)
wf = fcisolver.kernel(mf.mo_coeff)

mol_large = gto.Mole()
mol_large.atom = 'H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6'
mol_large.basis = 'sto-3g'
mol_large.build()

mf_large = scf.RHF(mol_large).run()

C = symm.inter_mo_transform(mol, mol_large, mf.mo_coeff, mf_large.mo_coeff)
wf_large_from_small = np.dot(wf, np.dot(C, C.T))

fcisolver_large = fci.FCISolver(mol_large)
wf_large_direct = fcisolver_large.kernel(mf_large.mo_coeff)

assert np.linalg.norm(wf_large_from_small - wf_large_direct) < 1e-10

# Transform the FCI wavefunction using a different method
wf_large_from_small_symmetrized = symm.symmetrize_wavefunction(mol_large, wf_large_from_small)

assert np.linalg.norm(wf_large_from_small_symmetrized - wf_large_direct) < 1e-10
```