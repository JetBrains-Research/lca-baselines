```python
import numpy as np
from pyscf import gto, scf, mp

mol = gto.Mole()
mol.atom = '''
    O 0.0 0.0 0.0
    H 0.757 0.586 0.0
    H -0.757 0.586 0.0
'''
mol.basis = 'sto-3g'
mol.build()

coords = np.random.rand(10, 3)
charges = np.random.rand(10)

def calculate_force(mol, coords, charges):
    # Calculate force from QM region on MM particles
    qm_mol = gto.Mole()
    qm_mol.atom = '''
        O 0.0 0.0 0.0
        H 0.757 0.586 0.0
        H -0.757 0.586 0.0
    '''
    qm_mol.basis = 'sto-3g'
    qm_mol.build()
    
    qm_scf = scf.RHF(qm_mol)
    qm_scf.kernel()
    
    qm_density = qm_scf.make_rdm1()
    
    # Calculate force from HF electron density
    hf_force = np.zeros((10, 3))
    for i in range(10):
        for j in range(3):
            hf_force[i, j] = np.sum(qm_density * charges[i] * (coords[:, j] - qm_mol.atom_coords()[:, j]))
    
    # Calculate force from MP2 electron density with orbital response
    mp2 = mp.MP2(qm_scf)
    mp2.kernel()
    
    mp2_rdm1 = mp2.make_rdm1()
    
    mp2_force = np.zeros((10, 3))
    for i in range(10):
        for j in range(3):
            mp2_force[i, j] = np.sum(mp2_rdm1 * charges[i] * (coords[:, j] - qm_mol.atom_coords()[:, j]))
    
    return hf_force, mp2_force

hf_force, mp2_force = calculate_force(mol, coords, charges)
```