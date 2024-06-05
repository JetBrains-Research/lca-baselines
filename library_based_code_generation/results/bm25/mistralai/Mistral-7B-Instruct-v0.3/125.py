 Here is the Python code based on your instructions using the pyscf library:

```python
from pyscf import gto, scf, cc, symm, ao2mo, df

# Define the cell
atom = 'H 0 0 0; F 0 0 a/2 0; F 0 0 -a/2 0'
basis = '6-31g'
pseudo = 'sto-3g'
a = 4.0
unit = 'Bohr'
verbose = 5

cell = gto.Cell()
cell.build(atom, a_angstrom=a, basis=basis, pseudo=pseudo, spin=0, verbose=verbose)

# Perform KHF and KMP2 calculations with 2x2x2 k-points
cell.set_kpts([2, 2, 2])
mf_khf = scf.RHF(cell).run()
mp2 = cc.CCSCF(mf_khf)
mp2.mp2()
kmp2_energy = mp2.energy()
print('KMP2 energy per unit cell (2x2x2 k-points):', kmp2_energy)

# Perform KHF and KMP2 calculations for a single k-point calculation
cell.set_kpts([1, 1, 1])
mf_khf_single = scf.RHF(cell).run()
mp2_single = cc.CCSCF(mf_khf_single)
mp2_single.mp2()
kmp2_energy_single = mp2_single.energy()
print('KMP2 energy per unit cell (single k-point):', kmp2_energy_single)

# Perform a single k-point calculation using the RHF method
rhf = scf.RHF(cell).run()

# Generate the first and second order reduced density matrices
rdm1 = rhf.make_rdm1()
rdm2 = rhf.make_rdm2()

# Calculate the total energy based on these matrices
total_energy = rhf.energy_ele() + df.kernel(cell, rdm1, rdm2)
print('RMP2 energy per unit cell at the k-point:', total_energy)

# Convert the RHF object to UHF and GHF objects
uhf = rhf.to_uhf()
ghf = rhf.to_ghf()

# Perform UMP2 and GMP2 calculations respectively
ump2 = cc.UCCSD(uhf).run()
ump2.mp2()
gmp2 = cc.GCCSD(ghf).run()
gmp2.mp2()

# Generate the first and second order reduced density matrices
ump2_rdm1 = ump2.make_rdm1()
ump2_rdm2 = ump2.make_rdm2()
gmp2_rdm1 = gmp2.make_rdm1()
gmp2_rdm2 = gmp2.make_rdm2()

# Calculate the total energy based on the MP2 density matrices
ump2_total_energy = ump2.energy() + df.kernel(cell, ump2_rdm1, ump2_rdm2)
gmp2_total_energy = gmp2.energy() + df.kernel(cell, gmp2_rdm1, gmp2_rdm2)

print('UMP2 energy per unit cell at the k-point:', ump2_total_energy)
print('GMP2 energy per unit cell at the k-point:', gmp2_total_energy)
```

This code creates a cell, performs KHF and KMP2 calculations with 2x2x2 k-points, repeats the calculations for a single k-point, performs a single k-point calculation using the RHF method, and calculates the total energy based on the RMP2 density matrices. It also converts the RHF object to UHF and GHF objects, performs UMP2 and GMP2 calculations, and calculates the total energy based on the MP2 density matrices. The energies are printed at the k-point and the unit cell level.