  import pyscf
import numpy as np

# Define molecule using pyscf
mol = pyscf.gto.M(
    atom='H 0.75 0.75 0.75; H 0.75 0.75 1.25; H 0.75 1.25 0.75; H 0.75 1.25 1.25; H 1.25 0.75 0.75; H 1.25 0.75 1.25; H 1.25 1.25 0.75; H 1.25 1.25 1.25',
    basis='6-31g'
)

# Generate random coordinates and charges for MM particles
n_particles = 100
coords = np.random.rand(n_particles, 3)
charges = np.random.rand(n_particles)

# Define function to calculate force
def calculate_force(coords, charges):
    # Calculate electron density from HF
    hf_density = pyscf.scf.RHF(mol).density()
    hf_energy = pyscf.scf.RHF(mol).kernel()

    # Calculate force from HF electron density
    hf_force = -np.einsum('ij,ji->i', hf_density, np.gradient(hf_energy))

    # Calculate MP2 electron density
    mp2_density = pyscf.mp.MP2(mol).kernel()
    mp2_energy = pyscf.mp.MP2(mol).kernel()

    # Calculate force from MP2 electron density
    mp2_force = -np.einsum('ij,ji->i', mp2_density, np.gradient(mp2_energy))

    # Calculate force from QM atoms and MM particles
    qm_force = np.zeros(n_particles)
    for i in range(mol.nao_nr()):
        qm_force += np.einsum('ij,ji->i', mol.intor('cint1e_r_sph', comp=3, aosym='s2', shls_slice=(0, mol.nbas, 0, mol.nbas)), np.gradient(mol.mo_coeff[:, i]))

    # Calculate force from electron density and MM particles
    force = hf_force + mp2_force + qm_force

    return force

# Calculate force from HF electron density and verify it
force = calculate_force(coords, charges)
print(force)

# Calculate force from MP2 electron density and verify it
force = calculate_force(coords, charges)
print(force)

# Calculate reduced density matrix with orbital response
rdm1 = pyscf.mp.make_rdm1(mol, mp2_density)

# Calculate force from MP2 electron density (including orbital response)
force = calculate_force(coords, charges)
print(force)

# Verify force from MP2 electron density (including orbital response)
force = calculate_force(coords, charges)
print(force)