 Here is a simplified Python code example using the pyscf library to calculate the triplet and quintet energy gaps of an Iron-Porphyrin molecule. Note that this code assumes you have a properly formatted molecular structure and basis set.

```python
from pyscf import gto, scf, dmrg, ccsd, nevpt2
from pyscf.dft.meth_hcth import HCTH
from pyscf.lib.libdf import make_pi_system_orbitals
from pyscf.symm.groups import SpaceGroup

# Define the molecular structure and basis set
mol = gto.Mole(
    atom='Fe H 1.95 0.00 0.00; N 2.00 0.00 0.00; N 0.00 2.00 0.00; N 0.00 0.00 2.00; H 1.95 2.00 0.00; H 1.95 0.00 2.00; H 2.00 1.95 0.00; H 2.00 0.00 1.95; H 0.00 2.00 1.95; H 0.00 1.95 2.00',
    basis='6-31g',
    spin=4,
    symmetry='D4h'
)

# Define the active space
active = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
frozen = [(i, j) for i, j in enumerate(active) if i % 2 == 0]

# Define the DMET active space
dmet_active = SpaceGroup(mol.symmetry, active, frozen)
dmet_active.enlarge_space()
dmet_active.symmetrize_space()

# Define the auxiliary space
aux_space = dmet_active.entire_space()
aux_space = aux_space[dmet_active.index_frozen_active(aux_space)]

# Define the DMET functionals
dmet_ccsd = dmrg.CCSD(mol, active=dmet_active, aux_space=aux_space)
dmet_nevpt2 = dmrg.NEVPT2(mol, active=dmet_active, aux_space=aux_space)

# Calculate the quintet energy
mol.spin = 5
quintet_ccsd_energy = dmet_ccsd.kernel()

# Calculate the triplet energy
mol.spin = 1
triplet_ccsd_energy = dmet_ccsd.kernel()

# Calculate the quintet energy using NEVPT2
quintet_nevpt2_energy = dmet_nevpt2.kernel()

# Calculate the triplet energy using NEVPT2
triplet_nevpt2_energy = dmet_nevpt2.kernel()

# Calculate the energy gaps
quintet_energy_gap = quintet_nevpt2_energy - quintet_ccsd_energy
triplet_energy_gap = triplet_nevpt2_energy - triplet_ccsd_energy

# Define the pi system orbitals
pi_system = make_pi_system_orbitals(mol, active)

# Calculate the molecular orbitals
mo_coeff = mol.mo_coeff()
mo_energy = mol.mo_energy()

# Extract the active space orbitals
active_mo_coeff = mo_coeff[:, dmet_active.active_orbs]
active_mo_energy = mo_energy[dmet_active.active_orbs]

# Output the active space orbitals to molden format
from pyscf.lib.molden import MoldenFile
MoldenFile(
    'active_space_orbitals.molden',
    title='Active Space Orbitals of Iron-Porphyrin',
    atoms=mol.atom,
    basis=mol.basis,
    coords=mol.cartcoords,
    occ=active_mo_coeff.T.diagonal(),
    e_occ=active_mo_energy,
    unocc=active_mo_coeff.T[:, dmet_active.virtual_orbs].diagonal(),
    e_unocc=active_mo_energy[len(dmet_active.active_orbs):]
)

print(f'Quintet Energy Gap: {quintet_energy_gap} Hartrees')
print(f'Triplet Energy Gap: {triplet_energy_gap} Hartrees')
```

This code calculates the triplet and quintet energy gaps using DMRG-CASSCF and DMRG-NEVPT2 methods from the pyscf library. It also outputs the active space orbitals to a molden file.