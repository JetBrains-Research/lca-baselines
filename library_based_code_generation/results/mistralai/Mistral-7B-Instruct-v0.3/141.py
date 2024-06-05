 Here is a simplified Python code using pyscf and scipy libraries that demonstrates the calculation of forces on MM particles from QM region. Note that this is a basic example and does not include all the complexities of a full QM/MM simulation.

```python
import numpy as np
from scipy.sparse import csr_matrix
import pyscf

# Define molecule
mol = pyscf.Molecule.from_xyz('molecule.xyz')

# Generate random coordinates and charges for MM particles
num_mm_atoms = 5
mm_coords = np.random.uniform(-10, 10, (num_mm_atoms, 3))
mm_charges = np.random.uniform(-1, 1, num_mm_atoms)

# Add MM particles to molecule
for i, coord in enumerate(mm_coords):
    mol.add_atom('H', coord[0], coord[1], coord[2], charge=mm_charges[i])

# Define function to calculate force on MM particles
def calculate_force(mol, mm_coords, mm_charges):
    # Calculate HF electron density
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    rho = mf.make_rdm1()

    # Calculate force from HF electron density
    force = np.zeros((num_mm_atoms, 3))
    for i, coord in enumerate(mm_coords):
        force[i] = pyscf.lib.numint.ao2mo.density_gradient(rho, coord)

    # Calculate force from Hartree-Fock (HF) orbital gradients
    mf_grad = pyscf.gradients.hf(mol, mf)
    for i, coord in enumerate(mm_coords):
        force[i] += pyscf.lib.numint.ao2mo.density_gradient(mf_grad.mo_coeff, coord)

    return force

# Calculate force from HF electron density and verify it
hf_force = calculate_force(mol, mm_coords, mm_charges)
print("HF Force:", hf_force)

# Define function to make reduced density matrix (rdm1) with orbital response
def make_rdm1_with_response(mf, mo_coeff_response):
    rdm1 = csr_matrix((mf.mo_coeff**2).sum(axis=0))
    for i, coeff in enumerate(mo_coeff_response):
        rdm1 += csr_matrix((coeff**2).sum(axis=0))
    return rdm1

# Define function to calculate force from MP2 electron density (including orbital response)
def calculate_mp2_force(mol, mm_coords, mm_charges, mo_coeff_response):
    # Calculate MP2 electron density
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    mp2 = pyscf.mp2.MP2(mf)
    mp2.kernel()
    rho = make_rdm1_with_response(mf, mo_coeff_response)

    # Calculate force from MP2 electron density
    force = np.zeros((num_mm_atoms, 3))
    for i, coord in enumerate(mm_coords):
        force[i] = pyscf.lib.numint.ao2mo.density_gradient(rho, coord)

    return force

# Calculate force from MP2 electron density (including orbital response) and verify it
mo_coeff_response = np.random.normal(0, 0.1, mf.nelectron)
mp2_force = calculate_mp2_force(mol, mm_coords, mm_charges, mo_coeff_response)
print("MP2 Force:", mp2_force)
```

This code defines a molecule, generates random coordinates and charges for MM particles, and defines functions to calculate the force on MM particles from HF and MP2 electron densities. The HF force is calculated using both the electron density and orbital gradients, while the MP2 force is calculated using the electron density with orbital response. The orbital response is simulated by adding a random perturbation to the HF orbitals.