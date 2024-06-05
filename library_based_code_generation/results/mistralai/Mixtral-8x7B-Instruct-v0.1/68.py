 from pyscf import gto, scf, agf2, df

mol = gto.Mole()
mol.atom = [
    ['H', ( 0.0000,  0.0000,  0.0000)],
    ['H', ( 0.0000,  0.0000,  0.7348)],
    ['He', ( 0.0000,  0.0000,  1.4697)],
]
mol.basis = '6-31g'
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
energy = mf.kernel()

agf = agf2.AGF2(mf)
agf.auxbasis = 'weigend'
agf.direct = True
agf.conv_tol = 1e-12
agf.kernel()

ip, ea = agf.ionization_potential(nstates=6)
print("IPs:", ip)
print("EAs:", ea)

mo_dm = mf.make_rdm1()
dip_moment = mol.intor('int1e_dip_moment')
dip_moment_mo = reduce(numpy.dot, (mf.mo_coeff.T, dip_moment, mf.mo_coeff))
dip_moment_mo += mol.atom_charges * numpy.eye(3)
print("Dipole Moment:", dip_moment_mo)