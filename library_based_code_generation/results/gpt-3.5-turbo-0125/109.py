from pyscf.pbc import gto, scf, cc
import numpy as np

cell = gto.Cell()
cell.atom = '''
C 0. 0. 0.
C 1.42 1.42 1.42
'''
cell.basis = 'gth-szv'
cell.a = np.eye(3) * 3.5668
cell.build()

mf = scf.RHF(cell).run()

mycc = cc.CCSD(mf).run()

mycc_ip = cc.CCSD(mf).ipccsd()
mycc_ea = cc.CCSD(mf).eaccsd()

print("Mean-field energy difference (gamma/k-point): ", mycc.e_tot - mf.e_tot)
print("CCSD energy difference (gamma/k-point): ", mycc.e_corr - mycc.e_tot)
print("IP-EOMCCSD energy difference (gamma/k-point): ", mycc_ip[0] - mycc.e_tot)
print("EA-EOMCCSD energy difference (gamma/k-point): ", mycc_ea[0] - mycc.e_tot)