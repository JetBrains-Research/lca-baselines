from pyscf import gto, scf, dft, fci
import numpy as np
import random

# Define molecule
mol = gto.M(
atom='H 0 0 0; F 0 0 1.1',
basis='6-31g',
charge=0,
spin=0,
)

# Generate random coordinates and charges for MM particles
n_mm_particles = 10
mm_coords = np.random.rand(n_mm_particles, 3)
mm_charges = np.random.rand(n_mm_particles)

def calculate_hf_force(mol, mm_coords, mm_charges):
na, nb = mol.nelec
dm = np.eye(na) + np.eye(nb)
mf = scf.RHF(mol)
mf.kernel()
mf.with_df.init_df()
dm_hf = mf.make_rdm1()
return mf.extra_force(dm_hf, mol.atom_coords() + mm_coords, mol.atom_charges() + mm_charges)

hf_force = calculate_hf_force(mol, mm_coords, mm_charges)

def calculate_mp2_force_orbital_response(mol, mm_coords, mm_charges):
na, nb = mol.nelec
mf = scf.RHF(mol)
mf.kernel()
mf.with_df.init_df()

mp = dft.RMP2(mf)
mp.kernel()

dm1 = mp.make_rdm1()
t1 = mp.get_t1()

def orbital_response_rdm1(dm1, t1):
return np.dot(np.dot(t1.T, dm1), t1) + np.dot(np.dot(t1.T, t1), dm1 - np.eye(na + nb))

rdm1_orbital_response = orbital_response_rdm1(dm1, t1)
return mf.extra_force(rdm1_orbital_response, mol.atom_coords() + mm_coords, mol.atom_charges() + mm_charges)

mp2_force = calculate_mp2_force_orbital_response(mol, mm_coords, mm_charges)

print("HF force:", hf_force)
print("MP2 force with orbital response:", mp2_force)