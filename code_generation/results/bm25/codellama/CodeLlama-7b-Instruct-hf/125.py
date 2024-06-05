  Here is the Python code based on the given instructions:
```
from pyscf import gto, scf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.tools import kpoints

# Create a cell using the pyscf.pbc.gto.Cell() function
cell = pbcgto.Cell()
cell.atom = 'C 0.0 0.0 0.0; C 0.0 0.0 1.0'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '3.0 3.0 3.0'
cell.unit = 'B'
cell.verbosity = 0

# Perform KHF and KMP2 calculations with 2x2x2 k-points
kpts = kpoints.monkhorst_pack(2, 2, 2)
khf = pbcscf.KRHF(cell, kpts)
kmp2 = pbcscf.KMP2(cell, kpts)
khf.kernel()
kmp2.kernel()
print('KMP2 energy per unit cell:', kmp2.e_tot)

# Perform a single k-point calculation using the RHF method
rhf = scf.RHF(cell)
rhf.kernel()
print('RMP2 energy per unit cell:', rhf.e_tot)

# Generate the first and second order reduced density matrices
rdm1 = rhf.make_rdm1()
rdm2 = rhf.make_rdm2()

# Calculate the total energy based on the MP2 density matrices
mp2_energy = rhf.energy_mp2(rdm1, rdm2)
print('MP2 energy per unit cell:', mp2_energy)

# Convert the RHF object to UHF and GHF objects
uhf = rhf.convert_to_uhf()
ghf = rhf.convert_to_ghf()

# Perform UMP2 and GMP2 calculations with the UHF and GHF objects
umf2 = uhf.UMP2()
gmf2 = ghf.GMP2()
umf2.kernel()
gmf2.kernel()
print('UMP2 energy per unit cell:', umf2.e_tot)
print('GMP2 energy per unit cell:', gmf2.e_tot)
```
Note that this code uses the `pyscf` library, which is a Python interface to the Quantum Chemistry Framework (Q-Chem) software package. The `pyscf.pbc` module provides a set of tools for performing quantum chemistry calculations on periodic systems. The `pyscf.pbc.gto` module provides a set of tools for creating and manipulating periodic cell objects. The `pyscf.pbc.scf` module provides a set of tools for performing quantum chemistry calculations on periodic systems. The `pyscf.pbc.tools` module provides a set of tools for performing various tasks related to quantum chemistry calculations on periodic systems.