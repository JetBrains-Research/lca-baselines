 ```python
import numpy as np
from pyscf import gto, scf, fci

# Task 1: Orbital transformation of FCI wavefunction
mol1 = gto.Mole()
mol1.atom = 'H 0 0 0; H 0 0 1.2'
mol1.basis = 'sto-3g'
mol1.build()

mol2 = gto.Mole()
mol2.atom = 'H 0 0 0; He 0 0 1.2'
mol2.basis = 'sto-3g'
mol2.build()

mf1 = scf.RHF(mol1).run()
mf2 = scf.RHF(mol2).run()

fcimol1 = fci.FCI(mf1).run()
fcimol2 = fci.FCI(mf2).run()

c1 = fcimol1.ci
c2 = fcimol2.ci

c1_transformed = np.dot(c1, mol1.intor('int1e_ovlp_sph'))
c1_transformed = np.dot(mol2.intor('int1e_ovlp_sph'), c1_transformed)

diff1 = np.linalg.norm(c1_transformed - c2)
print(f'Difference between original and transformed wavefunctions: {diff1:.8f}')

# Task 2: Expanding FCI wavefunction to a larger orbital space
mol3 = gto.Mole()
mol3.atom = 'H 0 0 0; He 0 0 1.2; H 0 1 1.2; H 1 0 1.2'
mol3.basis = 'sto-3g'
mol3.build()

mf3 = scf.RHF(mol3).run()
fcimol3 = fci.FCI(mf3).run()

c3 = fcimol3.ci

c1_expanded = np.dot(c1, mol1.intor('int1e_ovlp_sph'))
c1_expanded = np.dot(mol3.intor('int1e_ovlp_sph'), c1_expanded)

diff2 = np.linalg.norm(c1_expanded - c3)
print(f'Difference between expanded and FCI wavefunctions: {diff2:.8f}')

# Transform the FCI wavefunction using a different method
c1_transformed_alt = np.linalg.solve(mol2.intor('int1e_ovlp_sph'), np.dot(mol1.intor('int1e_ovlp_sph'), c1))

diff3 = np.linalg.norm(c1_transformed_alt - c2)
print(f'Difference between transformed (alternative method) and original wavefunctions: {diff3:.8f}')
```