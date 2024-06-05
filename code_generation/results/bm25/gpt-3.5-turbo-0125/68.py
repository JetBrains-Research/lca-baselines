import pyscf

mol = pyscf.gto.Mole()
mol.atom = '''
H 0.0 0.0 0.0
H 0.0 0.0 0.74
'''
mol.basis = 'sto-3g'
mol.build()

mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-8
mf.auxbasis = 'cc-pVDZ'
mf.kernel()

agf2 = pyscf.agf2.AGF2(mf)
agf2.conv_tol = 1e-8
agf2.kernel()

ip, ea = agf2.ip_ea(nroots=3)
print("Ionization Potentials:", ip)
print("Electron Affinities:", ea)

dm_mo = mf.make_rdm1()
dipole_integrals = mf.dip_moment(unit='au', verbose=0)
dipole_moment = pyscf.tools.mo_dipole(mol, dipole_integrals, dm_mo)
print("Calculated Dipole Moment:", dipole_moment)