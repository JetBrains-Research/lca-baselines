from pyscf import gto, scf, mcscf, dmrgscf
from pyscf.tools import molden

mol = gto.M(
    atom = 'Fe 0 0 0; N 0 0 2.1',
    basis = 'ccpvdz',
    symmetry = True,
)

# Define active space
nelec = (18, 16)
norb = 20
active_space = {'Fe': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                'N': [16, 17, 18, 19]}

# DMRG-CASSCF calculation for quintet state
mc = dmrgscf.DMRGSCF(mol)
mc.fcisolver.nroots = 1
mc.fcisolver.spin = 4
mc.fcisolver.singlet = False
mc.fcisolver.twopdm = True
mc.fcisolver.output_file = 'quintet.out'
mc.active_space = active_space
mc.kernel()

# DMRG-CASSCF calculation for triplet state
mc = dmrgscf.DMRGSCF(mol)
mc.fcisolver.nroots = 1
mc.fcisolver.spin = 2
mc.fcisolver.singlet = False
mc.fcisolver.twopdm = True
mc.fcisolver.output_file = 'triplet.out'
mc.active_space = active_space
mc.kernel()

# Output active space orbitals to molden format
molden.from_mo(mol, 'active_space.molden', mc.mo_coeff[:, :norb])