import pyscf
from pyscf import gto, scf, mcscf, dmrgscf, fci, mp, lo, molden
from pyscf.dmrgscf import settings

mol = gto.M(atom='Fe 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol).run()

# Define the active space
nelec = (8, 6)
norb = 12
active_space = {'Fe': [3, 4, 5, 6, 7, 8, 9, 10], 'N': [11, 12]}
cas = mcscf.CASSCF(mf, norb, nelec)
cas.fcisolver = dmrgscf.DMRGCI(mol)
cas.fcisolver.dmrg_options['output_root'] = 'FePorphyrin'
cas.fcisolver.dmrg_options['schedule'] = settings.scheduleSweeps(0, 10)
cas.fcisolver.dmrg_options['maxM'] = 1000
cas.fcisolver.dmrg_options['tol'] = 1e-8
cas.fcisolver.dmrg_options['num_thrds'] = 4
cas.fcisolver.dmrg_options['chkfile'] = 'FePorphyrin.chk'
cas.fcisolver.dmrg_options['maxIter'] = 20
cas.fcisolver.dmrg_options['verbose'] = 0
cas.active_space = active_space

# Calculate quintet energy
cas.spin = 2
cas.nroots = 1
cas.kernel()

# Calculate triplet energy
cas.spin = 1
cas.nroots = 1
cas.kernel()

# Output active space orbitals to molden format
mo_coeff = cas.mo_coeff
molden.from_mo(mol, 'FePorphyrin.molden', mo_coeff, occ=cas.ncore+cascas.ncas)