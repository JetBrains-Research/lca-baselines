 from pyscf import gto, scf, agf2, dft

mol = gto.Mole()
mol.atom = [['H', ( 0., 0., 0.)],
 ['H', ( 0., 0., 0.54)]]
mol.basis = '6-31g'
mol.spin = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

agf2_mf = agf2.AGF2(mf)
agf2_mf.conv_tol = 1e-12
agf2_mf.with_df = True
agf2_mf.df.auxbasis = 'weigend'
agf2_mf.kernel()

ip, ea = agf2_mf.get_ionization_potential(nstates=3, threshold=1e-8)
print('First three IPs:', ip)
print('First three EAs:', ea)

dm = mf.make_rdm1()
dip_mat = agf2_mf.get_dipole_moment(mol, dm)
dip_nuc = mol.intor('int1e_nuc_spinor', comp=3)
dipole_moment = dm.dot(dip_mat) + dip_nuc
print('Dipole moment:', dipole_moment)