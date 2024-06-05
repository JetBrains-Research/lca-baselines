 ```python
import numpy as np
import pyscf
from pyscf import gto, scf, dft, grad

# Define a molecule
mol = gto.M(
    atom='H 0 0 0; H 0 0 1.2',
    basis='sto-3g'
)

# Generate random coordinates and charges for MM particles
np.random.seed(1)
mm_coords = np.random.rand(10, 3)
mm_charges = np.random.rand(10) - 0.5

# Define a function to calculate the force
def calculate_force(density, mol, mm_coords, mm_charges):
    au2bohr = 0.52917721092
    natoms = mol.nao
    nmm = len(mm_coords)
    vmm = np.zeros((nmm, 3))

    # Interaction between QM atoms and MM particles
    for i, coord in enumerate(mm_coords):
        dist = np.linalg.norm(mol.coord * au2bohr - coord)
        vmm[i] = mm_charges[i] * density.sum(axis=0) / dist**3

    # Interaction between electron density and MM particles
    for i in range(natoms):
        for j in range(nmm):
            dist = np.linalg.norm(mol.coord[i] * au2bohr - mm_coords[j])
            vmm[j] += mol.intor_symmetric('int1e_aij(i,j,j)', i, j) * mm_charges[j] / dist**3

    return vmm * au2bohr

# Calculate the force from Hartree-Fock (HF) electron density
mf = scf.RHF(mol)
mf.kernel()
density = mf.make_rdm1()
force_hf = calculate_force(density, mol, mm_coords, mm_charges)

# Consider the response of HF orbitals in the analytical gradients for post-HF methods
mf_mp2 = dft.RKS(mol).density_fit()
mf_mp2.xc = 'mp2'
mf_mp2.kernel()
mo_coeff = mf_mp2.mo_coeff

# Define a function to make the reduced density matrix (rdm1) with orbital response
def make_rdm1_orbital_response(mo_coeff, dm1_hf):
    nao = mo_coeff.shape[0]
    mo_coeff = np.asarray(mo_coeff, order='C')
    dm1_orbital_response = np.zeros((nao, nao))
    for i in range(nao):
        for j in range(nao):
            dm1_orbital_response[i, j] = np.einsum('ij,ji->', mo_coeff[:, i], mo_coeff[:, j]) * (dm1_hf[i, j] - dm1_hf[j, i])
    return dm1_orbital_response

rdm1_orbital_response = make_rdm1_orbital_response(mo_coeff, density)

# Calculate the force from MP2 electron density (including orbital response)
grad_mp2 = grad.RHF(mf_mp2)
dip_grad_mp2 = grad_mp2.kernel(rdm1_orbital_response)
force_mp2 = np.zeros((len(mm_coords), 3))
for i, coord in enumerate(mm_coords):
    force_mp2[i] = -dip_grad_mp2[i]

# Verify the forces
print("Force from HF: ", force_hf)
print("Force from MP2: ", force_mp2)
```