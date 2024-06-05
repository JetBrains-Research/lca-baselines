 Here is a simplified Python code example using the pyscf library to calculate the triplet and quintet energy gaps of an Iron-Porphyrin molecule. Note that this is a complex task and requires a detailed molecular structure and appropriate basis sets. Also, the DMET active space definition and the specific methods (DMRG-CASSCF and DMRG-NEVPT2) might need adjustments based on your system.

```python
from pyscf import gto, scf, dmrg, ao2mo, mol, molden

# Define the molecular geometry
atom = [
    ['Fe', (0, 0, 0)],
    ['N', (0, 0, 1.5)],
    ['N', (0, 0, -1.5)],
    # Add more N atoms for the porphyrin structure
    # ...
]

mol = gto.Molecule(atom=atom)
mol.basis = '6-31g'  # Adjust basis set as needed

# Define the active space
active_space = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]  # Fe double d-shell, 4s shell, and ligand N 2pz orbitals

# Define the DMET functionals and methods
dmet_functional = 'dmet_r12'
casscf_method = scf.CASSCF(mol, active_orbs=active_space, chkfile='casscf.chk')
nevpt2_method = dmrg.NEVPT2(casscf_method, active_orbs=active_space, chkfile='nevpt2.chk')

# Calculate the quintet and triplet energies
casscf_results = casscf_method.run(conv_tol=1e-12)
nevpt2_results = nevpt2_method.run(nroots=2, nevpt2_nmax=4, conv_tol=1e-12)

quintet_energy = nevpt2_results.e_tot[0]
triplet_energy = nevpt2_results.e_tot[1]

print(f'Quintet Energy: {quintet_energy} Hartrees')
print(f'Triplet Energy: {triplet_energy} Hartrees')
print(f'Energy gap: {quintet_energy - triplet_energy} Hartrees')

# Output the active space orbitals to molden format
mo_coeff = casscf_results.mo_coeff
mo_energy = casscf_results.mo_energy

molden.MoldenFile(filename='active_space.molden').from_mo(mo_coeff, mo_energy, mol)
```

This code assumes that you have the pyscf library installed and that the molecular geometry and basis set are appropriate for your Iron-Porphyrin molecule. Adjust the code as needed to fit your specific system.