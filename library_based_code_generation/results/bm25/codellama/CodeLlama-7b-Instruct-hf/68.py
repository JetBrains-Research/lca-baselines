  from pyscf import gto, scf

mol = gto.M(atom='H 0.75 0.75 0.75; H 0.75 0.75 1.25; H 0.75 1.25 0.75; H 0.75 1.25 1.25; H 1.25 0.75 0.75; H 1.25 0.75 1.25; H 1.25 1.25 0.75; H 1.25 1.25 1.25', basis='6-31g')

mf = scf.RHF(mol)
mf.conv_tol = 1e-9
mf.kernel()

agf2 = scf.AGF2(mf)
agf2.conv_tol = 1e-9
agf2.kernel()

print(agf2.mo_energy[0:3])
print(agf2.mo_energy_a[0:3])

dm = agf2.make_rdm1()
dipole = scf.dipole(mol, dm)
print(dipole)

mo_basis_dm = agf2.transform_rdm(dm)
dipole_mo = scf.dipole(mol, mo_basis_dm)
print(dipole_mo)

nuc_dm = scf.add_nuc(mol, dm)
nuc_dipole = scf.dipole(mol, nuc_dm)
print(nuc_dipole)