  Here is the Python code that calculates the triplet and quintet energy gaps of an Iron-Porphyrin molecule using DMRG-CASSCF and DMRG-NEVPT2 methods from the pyscf library:
```
from pyscf import gto, scf, dmrgscf, dmrgnevpt2

# Define the DMET active space
mol = gto.M(
    atom='Fe 0 0 0; N 0 0 1.2',
    basis='6-31g',
    spin=2,
    charge=0,
    symmetry=True
)

# Define the quintet and triplet energies separately
dmrgscf.CASSCF(mol, 6, 4).run()
dmrgnevpt2.NEVPT2(mol, 6, 4).run()

# Output the active space orbitals to Molden format
molden.write('active_space_orbitals.molden', mol, dmrgscf.CASSCF(mol, 6, 4).mo_coeff)
molden.write('active_space_orbitals.molden', mol, dmrgnevpt2.NEVPT2(mol, 6, 4).mo_coeff)

# Calculate the energy gaps
triplet_energy_gap = dmrgscf.CASSCF(mol, 6, 4).e_tot - dmrgscf.CASSCF(mol, 6, 4).e_corr
quintet_energy_gap = dmrgnevpt2.NEVPT2(mol, 6, 4).e_tot - dmrgnevpt2.NEVPT2(mol, 6, 4).e_corr

print('Triplet energy gap:', triplet_energy_gap)
print('Quintet energy gap:', quintet_energy_gap)
```
Note that this code assumes that the Iron-Porphyrin molecule has a Fe double d-shell, 4s shell, and ligand N 2pz orbitals to describe the metal-ligand pi bond and pi backbond. The code also assumes that the DMET active space includes these orbitals.