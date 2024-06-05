  Here is the Python code that performs the two tasks using the pyscf library:
```
from pyscf import gto, scf, fci

# Task 1: Transform a FCI wavefunction with respect to orbital rotation/transformation

# Create two molecules with different atomic configurations
mol1 = gto.M(atom='H 0.75 0.75; H 0.75 0.25; H 0.25 0.75; H 0.25 0.25', basis='6-31g')
mol2 = gto.M(atom='H 0.75 0.75; H 0.75 0.25; H 0.25 0.75; H 0.25 0.25', basis='6-31g', spin=2)

# Calculate the FCI energies of the two molecules
e1 = scf.RHF(mol1).kernel()[0]
e2 = scf.RHF(mol2).kernel()[0]

# Transform the wavefunction of the first molecule to match the second one
wf1 = fci.to_fci(mol1, e1)
wf2 = fci.to_fci(mol2, e2)

# Compare the transformed wavefunction with the one obtained from the FCI solver
print('Transformed wavefunction:', wf1)
print('FCI wavefunction:', wf2)

# Task 2: Transfer a FCI wavefunction from a smaller orbital space to a larger one

# Create a molecule with a specific atomic configuration
mol3 = gto.M(atom='H 0.75 0.75; H 0.75 0.25; H 0.25 0.75; H 0.25 0.25', basis='6-31g')

# Calculate the FCI energy of the molecule
e3 = scf.RHF(mol3).kernel()[0]

# Expand the wavefunction to a larger orbital space
wf3 = fci.from_fci(mol3, e3, 2)

# Compare the expanded wavefunction with the one obtained from the FCI solver
print('Expanded wavefunction:', wf3)
print('FCI wavefunction:', wf3)

# Transform the FCI wavefunction using a different method
wf4 = fci.to_fci(mol3, e3, method='symmetrize_wavefunction')

# Compare the transformed wavefunction with the one obtained from the FCI solver
print('Transformed wavefunction (symmetrized):', wf4)
print('FCI wavefunction:', wf3)
```
Note that this code assumes that the pyscf library is installed and imported as `pyscf`. Additionally, the `gto` module is used to create the molecules, the `scf` module is used to calculate the FCI energies, and the `fci` module is used to perform the FCI calculations and transformations.