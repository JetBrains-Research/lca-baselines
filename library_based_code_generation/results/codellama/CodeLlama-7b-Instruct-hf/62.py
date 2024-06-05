  from pyscf import gto, scf, dmrgscf, molden

# Define the DMET active space
mol = gto.M(
    atom='Fe 0 0 0; N 0 0 1.2',
    basis='6-31g',
    spin=2,
    charge=0,
    verbose=0
)

# Define the DMRG-CASSCF method
dmrg_casscf = dmrgscf.DMRGCASSCF(mol, 8, 16, 1)

# Define the DMRG-NEVPT2 method
dmrg_nevpt2 = dmrgscf.DMRGNEVPT2(mol, 8, 16, 1)

# Calculate the quintet energy gap
quintet_energy = dmrg_casscf.kernel()[0]

# Calculate the triplet energy gap
triplet_energy = dmrg_nevpt2.kernel()[0]

# Output the active space orbitals to Molden format
molden.from_mo(mol, dmrg_casscf.mo_coeff, 'quintet.molden')
molden.from_mo(mol, dmrg_nevpt2.mo_coeff, 'triplet.molden')

print('Quintet energy gap:', quintet_energy)
print('Triplet energy gap:', triplet_energy)