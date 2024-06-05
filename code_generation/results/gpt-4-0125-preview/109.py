```python
from pyscf.pbc import gto, scf, cc
from pyscf import cc as mol_cc
import numpy as np

# Define a supercell
cell = gto.Cell()
cell.atom = '''
He 0 0 0
He 1 0 0
'''
cell.basis = 'cc-pvdz'
cell.a = np.eye(3) * 3
cell.verbose = 4
cell.build()

# Gamma point calculation
mf_gamma = scf.RHF(cell).run()
cc_gamma = mol_cc.CCSD(mf_gamma).run()
e_ip_gamma, _ = cc_gamma.ipccsd(nroots=1)
e_ea_gamma, _ = cc_gamma.eaccsd(nroots=1)

# K-point calculations
kpts = cell.make_kpts([2, 2, 2])
mf_k = scf.KRHF(cell, kpts=kpts).run()
cc_k = cc.KCCSD(mf_k).run()
e_ip_k, _ = cc_k.ipccsd(nroots=1)
e_ea_k, _ = cc_k.eaccsd(nroots=1)

# Calculate differences
delta_mf = mf_gamma.e_tot - mf_k.e_tot
delta_ccsd = cc_gamma.e_tot - cc_k.e_tot
delta_ip = e_ip_gamma - e_ip_k[0]
delta_ea = e_ea_gamma - e_ea_k[0]

# Print differences
print(f"Delta MF: {delta_mf}")
print(f"Delta CCSD: {delta_ccsd}")
print(f"Delta IP-EOMCCSD: {delta_ip}")
print(f"Delta EA-EOMCCSD: {delta_ea}")
```