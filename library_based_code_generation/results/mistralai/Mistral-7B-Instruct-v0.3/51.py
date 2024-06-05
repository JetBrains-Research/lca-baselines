 Here is the Python code using pyscf library for the tasks you've described:

```python
from pyscf import gto, scf, cc, symm, ao2mo
from pyscf.lib import numpy as np

# Define atomic coordinates, basis, pseudopotential, lattice vectors, and unit
atom = [['C', (0, 0, 0), 'sto3g']]
mol = gto.Mole(atom=atom, a0=3.5, basis='sto3g', pw=gto.PW(pseudos='gth-pbe-sol1'))
mol.build()

# Perform KHF and KCCSD calculations with 2x2x2 k-points
mol.spin = 0
kpts = mol.make_kpts([2, 2, 2])
ehf_2x2x2 = scf.RHF(mol, kpts=kpts).run()
eccsd_2x2x2 = cc.CCSD(ehf_2x2x2).run()
print("Total energy per unit cell (2x2x2 k-points):", eccsd_2x2x2.e_tot / len(kpts))

mol.kpts = [(0, 0, 0)]
ehf_1k = scf.RHF(mol).run()
eccsd_1k = cc.CCSD(ehf_1k).run()
print("Total energy per unit cell (single k-point):", eccsd_1k.e_tot)

# Perform RHF, RCCSD, and print total energy per unit cell at the k-point
ehf_1k_rhf = scf.RHF(mol).run()
rccsd_1k = cc.RCCSD(ehf_1k_rhf).run()
print("Total energy per unit cell (RHF, RCCSD):", rccsd_1k.e_tot)

rccsd_1k_energy = ao2mo.restore(rccsd_1k.mo_ccsd, mol)['E']
print("RCCSD energy:", rccsd_1k_energy)

# Perform UHF, UCCSD, and print total energy per unit cell at the k-point
mol.spin = 2
uhf_1k = scf.UHF(mol).run()
uccsd_1k = cc.UCCSD(uhf_1k).run()
print("Total energy per unit cell (UHF, UCCSD):", uccsd_1k.e_tot)

uccsd_1k_energy = ao2mo.restore(uccsd_1k.mo_ccsd, mol)['E']
print("UCCSD energy:", uccsd_1k_energy)

# Perform GHF, GCCSD, and print total energy per unit cell at the k-point
mol.symmetry = True
ghf_1k = symm.GHF(mol).run()
gccsd_1k = cc.GCCSD(ghf_1k).run()
print("Total energy per unit cell (GHF, GCCSD):", gccsd_1k.e_tot)

gccsd_1k_energy = ao2mo.restore(gccsd_1k.mo_ccsd, mol)['E']
print("GCCSD energy:", gccsd_1k_energy)
```

This code performs the tasks you've described, but it assumes that the 'gth-pbe-sol1' pseudopotential files are available in the working directory. You can replace 'gth-pbe-sol1' with the appropriate pseudopotential for carbon atoms if needed.