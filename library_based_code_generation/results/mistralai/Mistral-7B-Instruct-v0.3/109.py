 Here is a simplified example of how you might structure your code using PySCF. Note that this is a complex task and requires a good understanding of DFT and CC theory, as well as PySCF's API. Also, this code does not include error handling or optimization, and it assumes that you have a function `create_supercell` that creates a supercell from a given unit cell.

```python
import numpy as np
from pyscf import gto, scf, cc, mcscf

# Define the unit cell
atom = gto.Molecule()
atom.atom = 'H 0 0 0; H 0 0 a; H 0 a 0'.split()
atom.basis = 'sto-3g'
atom.spin = 1
atom.a = 4.0  # cell parameter

# Create a supercell
n_replicas = 3
supercell = create_supercell(atom, n_replicas)

# Define the K-point grid
kpts = [(1, 1, 1)]  # gamma point
kpts_grid = [np.array(kpts)]

# Define the Hartree-Fock function
mf = scf.RHF(supercell)
mf.kernel()

# Define the CCSD function
ccsd = cc.CCSD(mf)

# Define the IP-EOM-CCSD and EA-EOM-CCSD functions
ip_eom_ccsd = mcscf.CASCC(mf, ncas=2, nelec=4, symm=True, cholesky=True)
ea_eom_ccsd = mcscf.CASCC(mf, ncas=2, nelec=4, symm=True, cholesky=True, eom_cc='ea')

# Perform calculations at the gamma point
for method in [mf, ccsd, ip_eom_ccsd, ea_eom_ccsd]:
    method.kernel()
    energies_gamma = [method.e_tot]

# Perform K-point calculations
for kpts_ in kpts_grid:
    kpts[0] = kpts_[0]
    mf.kpts = kpts
    mf.kernel()

    for method in [ccsd, ip_eom_ccsd, ea_eom_ccsd]:
        method.kpts = kpts
        method.kernel()

        energies_kpt = [method.e_tot]

    energies_kpt_mean = np.mean(energies_kpt)
    energies_gamma_mean = np.mean(energies_gamma)
    differences = [energies_kpt_mean - energy for energy in energies_gamma]

    print(f'Method: RHF, Difference: {energies_kpt_mean - mf.e_tot}')
    print(f'Method: CCSD, Difference: {energies_kpt_mean - ccsd.e_tot}')
    print(f'Method: IP-EOM-CCSD, Difference: {energies_kpt_mean - ip_eom_ccsd.e_tot}')
    print(f'Method: EA-EOM-CCSD, Difference: {energies_kpt_mean - ea_eom_ccsd.e_tot}\n')
```

This code creates a supercell, performs Hartree-Fock and CC calculations at the gamma point, and then performs K-point calculations for the same system. It calculates the differences between gamma/k-point mean-field, ccsd, ip-eomccsd, and ea-eomccsd calculations and prints these differences.

Please note that this code does not include the `create_supercell` function, which you would need to implement based on your specific requirements. Also, the code assumes that the `ncas` and `nelec` parameters for the IP-EOM-CCSD and EA-EOM-CCSD functions are correctly set for your system. You may need to adjust these parameters based on your specific system and basis set.

Lastly, this code uses the Restricted Hartree-Fock (RHF) method, but you can change it to the Unrestricted Hartree-Fock (UHF) method if necessary.