```python
import numpy as np
from pyscf import gto, scf, grad, mp
from pyscf.grad import mp2

def generate_mm_particles(num_particles=5):
    np.random.seed(0)  # For reproducibility
    coordinates = np.random.rand(num_particles, 3) * 10  # Random coordinates in a 10x10x10 box
    charges = np.random.randint(1, 3, size=num_particles)  # Random charges between 1 and 2
    return coordinates, charges

def calculate_qm_mm_force(mol, mm_coords, mm_charges, method='HF'):
    if method == 'HF':
        mf = scf.RHF(mol)
        mf.kernel()
        dm = mf.make_rdm1()
    elif method == 'MP2':
        mf = scf.RHF(mol).run()
        mp2_grad = mp2.Gradients(mp.MP2(mf))
        dm = mp2_grad.make_rdm1()
    else:
        raise ValueError("Unsupported method")

    forces = np.zeros_like(mm_coords)
    for i, (coord, charge) in enumerate(zip(mm_coords, mm_charges)):
        for j in range(mol.natm):
            qm_charge = mol.atom_charge(j)
            qm_coord = np.array(mol.atom_coord(j))
            r = np.linalg.norm(coord - qm_coord)
            force_dir = (coord - qm_coord) / r
            forces[i] += qm_charge * charge / r**2 * force_dir  # Coulomb's law for QM-MM interaction

        # Interaction between electron density and MM particles
        for grid_coord in mol.grids.coords:
            r = np.linalg.norm(coord - grid_coord)
            if r > 1e-12:  # Avoid division by zero
                rho = mol.grids.weights.dot(dm)
                force_dir = (coord - grid_coord) / r
                forces[i] += rho * charge / r**2 * force_dir  # Interaction with electron density

    return forces

# Define a molecule
mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g')

# Generate MM particles
mm_coords, mm_charges = generate_mm_particles()

# Calculate force from HF electron density
hf_forces = calculate_qm_mm_force(mol, mm_coords, mm_charges, method='HF')
print("HF Forces:", hf_forces)

# Calculate force from MP2 electron density (including orbital response)
mp2_forces = calculate_qm_mm_force(mol, mm_coords, mm_charges, method='MP2')
print("MP2 Forces:", mp2_forces)
```