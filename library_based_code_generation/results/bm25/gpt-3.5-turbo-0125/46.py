import pyscf

# Create molecules with different atomic configurations
mol1 = pyscf.gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
mol2 = pyscf.gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')

# Calculate FCI energies for both molecules
fci_solver = pyscf.FCI(mol1)
fci_energy1 = fci_solver.kernel()
fci_solver.mol = mol2
fci_energy2 = fci_solver.kernel()

# Transform wavefunction of molecule 1 to match molecule 2
fci_solver1 = pyscf.FCI(mol1)
fci_solver2 = pyscf.FCI(mol2)
fci_solver1.ci = fci_solver1.to_fci(fci_solver2.ci)

# Transfer FCI wavefunction from smaller orbital space to larger one
mol = pyscf.gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
fci_solver = pyscf.FCI(mol)
fci_energy = fci_solver.kernel()

# Expand wavefunction to larger orbital space
fci_solver.expand_space()

# Compare transformed wavefunction with FCI solver wavefunction
is_close = pyscf.lib.fci.addons.symmetrize_wavefunction(fci_solver1.ci, fci_solver.ci)

# Transform FCI wavefunction using a different method
fci_solver1.ci = fci_solver1.from_fci(fci_solver2.ci)

# Compare results of different transformation methods
is_close2 = pyscf.lib.fci.addons.symmetrize_wavefunction(fci_solver1.ci, fci_solver2.ci)