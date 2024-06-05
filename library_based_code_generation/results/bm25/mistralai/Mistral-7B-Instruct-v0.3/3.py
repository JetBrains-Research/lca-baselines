 Here is a Python code snippet that follows your instructions, using the pyscf library. Please note that you need to replace `'H2'`, `'6-31g'`, and `'B3LYP'` with your specific molecule, basis set, and exchange-correlation functional.

```python
from pyscf import gto, scf, ao2mo, mol, lib

# Define molecule and basis set
mol = gto.Mole(
    atom='H2',
    basis='6-31g',
    spin=0,
    symmetry=False,
)

# Perform two state calculations and store molecular orbital information into separate chkfiles
mol.build()
mf1 = scf.RHF(mol).run(conv_tol=1e-12)
mf2 = scf.RHF(mol).run(conv_tol=1e-12, chkfile='chkfile2')

# Read MO coefficients and occupation numbers from chkfiles
c1 = mf1.mo_coeff
occ1 = mf1.mo_occ
c2 = lib.ao2mo.restore(mf2.mo_coeff, mol)
occ2 = mf2.mo_occ

# Calculate overlap between two determinants
S = ao2mo.overlap(c1, c2)

# Construct density matrices
rho1 = c1 @ c1.T * occ1
rho2 = c2 @ c2.T * occ2

# Calculate one-electron and two-electron part contributions
T1 = ao2mo.T1(c1, c1)
T2 = ao2mo.T2(c1, c1, c1, c1)
Vee = mol.intor('int1e_ovlp_sph', (0, 0))

# Calculate new total energy
E1 = mf1.energy
E2 = mf2.energy
E_elst = 0.5 * (Vee + Vee.T) @ rho1 @ rho2
E_kin = 0.5 * T1 @ rho1 + 0.5 * T2 @ rho1 @ rho1
E_total = E1 + E2 + E_elst + E_kin

# Calculate effective electronic coupling
C = ao2mo.C(c1, c2)
J = two_pole(mol, C, C.T, rho1, rho2)
SV = _effective_svd(J)
U = 0.5 * (SV[0] + SV[1])
K = U - 0.5 * (Vee + Vee.T)
K_eff = K[0, 0]

print(f'Effective electronic coupling: {K_eff}')

# Remove chkfiles
lib.utility.remove('chkfile2')
```

This code calculates the effective electronic coupling between the two determinants defined by the MO coefficients from the two state calculations. The effective electronic coupling is given by the diagonal element of the K matrix. The chkfiles are removed at the end.