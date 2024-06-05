 from pyscf import gto, scf, mp, dft, wannier90
from pyscf.wannier90 import wannier_plot as mcu

# Define unit cell
cell = gto.Cell()
cell.atom = '''
H 0 0 0; H 0 0.75 0.587
'''
cell.basis = 'gth-szv'
cell.unit = 'B'
cell.a = '''
0.00000000,   3.78419163,   3.78419163
3.78419163,   0.00000000,   3.78419163
3.78419163,   3.78419163,   0.00000000
'''

# Perform PBE calculation
mf = dft.PBE(cell)
mf.kernel()

# Save and load kks object
kks = mf.kks
wannier90.write_wannier90_hr(cell, kks, 'hr.win', mf.get_bands())
kks = wannier90._load_and_unpack('hr.win')

# Construct MLWFs
mlwf = wannier90.Wannier90(cell, kks)
mlwf.get_wannier_functions()

# Export MLWFs in xsf format for plotting
wannier90.export_unk(cell, kks, 'wannier_functions.xsf', 'wannier_functions')

# Export certain matrices and run wannier90 using these
wannier90.write_wannier90_hr(cell, kks, 'hr.win', mf.get_bands())
wannier90.write_wannier90_sr(cell, kks, 'sr.win', mf.get_bands())
wannier90.write_wannier90_mmn(cell, kks, 'mmn', mf.get_bands())
wannier90.write_wannier90_amn(cell, kks, 'amn', mf.get_bands())
wannier90.write_wannier90_eig(cell, kks, 'eig', mf.get_bands())
wannier90.write_wannier90_ucf(cell, kks, 'ucf', mf.get_bands())
wannier90.write_wannier90_umat(cell, kks, 'umat', mf.get_bands())
wannier90.write_wannier90_u0(cell, kks, 'u0', mf.get_bands())
wannier90.write_wannier90_uij(cell, kks, 'uij', mf.get_bands())
wannier90.write_wannier90_uprime(cell, kks, 'uprime', mf.get_bands())
wannier90.write_wannier90_uprimeij(cell, kks, 'uprimeij', mf.get_bands())
wannier90.write_wannier90_uprimeijab(cell, kks, 'uprimeijab', mf.get_bands())
wannier90.write_wannier90_j(cell, kks, 'j', mf.get_bands())
wannier90.write_wannier90_jij(cell, kks, 'jij', mf.get_bands())
wannier90.write_wannier90_jijab(cell, kks, 'jijab', mf.get_bands())
wannier90.write_wannier90_jk(cell, kks, 'jk', mf.get_bands())
wannier90.write_wannier90_jkab(cell, kks, 'jkab', mf.get_bands())
wannier90.write_wannier90_jkij(cell, kks, 'jkij', mf.get_bands())
wannier90.write_wannier90_jkijk(cell, kks, 'jkijk', mf.get_bands())
wannier90.write_wannier90_jkijab(cell, kks, 'jkijab', mf.get_bands())
wannier90.write_wannier90_jkabij(cell, kks, 'jkabij', mf.get_bands())
wannier90.write_wannier90_jkabcd(cell, kks, 'jkabcd', mf.get_bands())
wannier90.write_wannier90_r(cell, kks, 'r', mf.get_bands())
wannier90.write_wannier90_rr(cell, kks, 'rr', mf.get_bands())
wannier90.write_wannier90_rrr(cell, kks, 'rrr', mf.get_bands())
wannier90.write_wannier90_rrrr(cell, kks, 'rrrr', mf.get_bands())
wannier90.write_wannier90_rrrrr(cell, kks, 'rrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrr(cell, kks, 'rrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrr(cell, kks, 'rrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrr(cell, kks, 'rrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrr(cell, kks, 'rrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrr(cell, kks, 'rrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrr(cell, kks, 'rrrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrrr(cell, kks, 'rrrrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrrrr(cell, kks, 'rrrrrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrrrrr(cell, kks, 'rrrrrrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrrrrrr(cell, kks, 'rrrrrrrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrrrrrrr(cell, kks, 'rrrrrrrrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrrrrrrrr(cell, kks, 'rrrrrrrrrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrrrrrrrrr(cell, kks, 'rrrrrrrrrrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrrrrrrrrrr(cell, kks, 'rrrrrrrrrrrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrrrrrrrrrrr(cell, kks, 'rrrrrrrrrrrrrrrrrrrr', mf.get_bands())
wannier90.write_wannier90_rrrrrrrrrrrrrrrrrrrrr(cell, kks, 'rrrrrrrrrrrrrrrr