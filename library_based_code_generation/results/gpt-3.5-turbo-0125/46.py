import numpy as np
from pyscf import gto, scf, fci

# Task 1: Transform FCI wavefunction with respect to orbital rotation/transformation
mol1 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
mol2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')

mf1 = scf.RHF(mol1).run()
mf2 = scf.RHF(mol2).run()

cisolver = fci.FCI(mol1, mf1.mo_coeff)
e1, fcivec = cisolver.kernel()

cisolver = fci.FCI(mol2, mf2.mo_coeff)
e2, fcivec2 = cisolver.kernel()

transformed_fcivec = fci.addons.transform_ci_for_orbital_rotation(fcivec, mf1.mo_coeff, mf2.mo_coeff)

# Task 2: Transfer FCI wavefunction from smaller orbital space to larger one
mol3 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')

mf3 = scf.RHF(mol3).run()

cisolver = fci.FCI(mol3, mf3.mo_coeff)
e3, fcivec3 = cisolver.kernel()

mo_coeff_expanded = np.hstack((mf3.mo_coeff, np.zeros((mol3.nao_nr(), 1))))
transformed_fcivec_expanded = fci.addons.addons.transform_ci_for_orbital_rotation(fcivec3, mf3.mo_coeff, mo_coeff_expanded)

# Compare transformed wavefunction with FCI solver result
print(np.allclose(transformed_fcivec, fcivec2))
print(np.allclose(transformed_fcivec_expanded, fcivec3))

# Transform FCI wavefunction using a different method and compare results
transformed_fcivec_diff = fci.addons.transform_ci_for_orbital_rotation(fcivec, mf1.mo_coeff, mf2.mo_coeff, method='lowdin')
print(np.allclose(transformed_fcivec_diff, fcivec2))