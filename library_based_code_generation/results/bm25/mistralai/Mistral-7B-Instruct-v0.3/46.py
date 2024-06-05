 Here is a simplified Python code using pyscf library that performs the tasks you've described. Note that this code is a basic example and may need to be adjusted to fit your specific needs.

```python
from pyscf import gto, scf, symm, fci

# Define molecules
mol1 = gto.Molecule()
mol1.atom = 'H 0 0 0; H 0 0 1.4'
mol1.basis = 'sto-3g'
mol1.spin = 0

mol2 = gto.Molecule()
mol2.atom = 'H 0 0 0; H 0 0 2.8'
mol2.basis = 'sto-3g'
mol2.spin = 0

# Calculate FCI energies and wavefunctions
fci1 = StateSpecificFCISolver(mol1)
fci1.kernel()
E1 = fci1.e_tot
psi1 = fci1.mo_coeff

fci2 = StateSpecificFCISolver(mol2)
fci2.kernel()
E2 = fci2.e_tot

# Transform wavefunction of the first molecule to match the second one
symm_ops = symm.Symmetry(mol2)
psi1_sym = symm.project(psi1, symm_ops)
psi1_trans = symm.transform(psi1_sym, symm_ops.C)

# Expand the wavefunction to a larger orbital space
mol1_large = gto.Molecule()
mol1_large.atom = mol1.atom
mol1_large.basis = '6-31g'
mol1_large_fci = StateSpecificFCISolver(mol1_large)
mol1_large_fci.kernel()
psi1_large = mol1_large_fci.mo_coeff

# Compare transformed wavefunction with the one obtained from the FCI solver
diff = (psi1_trans - mol2_fci.mo_coeff)**2
print('Difference in wavefunctions:', diff.sum())

# Transform the FCI wavefunction using a different method
psi1_rotated = symm.rotate_mo(psi1, symm_ops.C)
diff_rotated = (psi1_rotated - mol2_fci.mo_coeff)**2
print('Difference in wavefunctions (rotation method):', diff_rotated.sum())
```

This code calculates the FCI energies and wavefunctions for two molecules with different atomic configurations (mol1 and mol2). It then transforms the wavefunction of the first molecule to match the second one using symmetry operations. After that, it expands the wavefunction to a larger orbital space and compares the transformed wavefunction with the one obtained from the FCI solver. Finally, it transforms the FCI wavefunction using a rotation method and compares the results with the previous transformation.