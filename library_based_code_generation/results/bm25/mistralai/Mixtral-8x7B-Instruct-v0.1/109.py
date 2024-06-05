 import numpy as np
from pyscf import gto, scf, cc, mp, df, fci
from pyscf.cc import rccsd
from pyscf.mp import mp2
from pyscf.fci import direct_spin0
from pyscf.tools import molden
from pyscf.prop import uccsd_symm
from pyscf.cc import ipccsd
from pyscf.cc import eaccsd

# Define the unit cell
cell = gto.M(
    atom = 'H 0 0 0; H 0 0 1.4',
    basis = '6-31g',
    a = np.eye(3)*1.4,
    verbose = 4,
    symmetry = 'd2h'
)

# Define the supercell
supercell = cell.supercell(2, 2, 2)

# Run molecular Hartree-Fock program using integrals between periodic gaussians
mf = scf.KRHF(supercell).run()

# Call a molecular CC method for gamma point calculation
mycc = cc.KCCSD(mf).run()

# Perform k-point calculations for the same system
kpts = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
mycc_kpts = cc.KCCSD(mf).newton(kpts)

# Calculate the differences between gamma/k-point mean-field, ccsd, ip-eomccsd, and ea-eomccsd calculations
print("Delta E(MP2):", mp2.kernel(supercell)[0] - mp.MP2(mf).kernel()[0])
print("Delta E(CCSD):", mycc.e_tot - mycc_kpts.e_tot)
print("Delta E(IP-EOMCCSD):", ipccsd.kernel(mycc)[0] - ipccsd.kernel(mycc_kpts)[0])
print("Delta E(EA-EOMCCSD):", eaccsd.kernel(mycc)[0] - eaccsd.kernel(mycc_kpts)[0])