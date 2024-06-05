 ```python
import pyscf
from pyscf import gto, scf, dmrgscf, mcscf, nesther, tools

# Define molecule
mol = gto.M(
    atom='FeTPP 0 0 0; N 0 0 5.282',
    basis='def2-tzvp',
    spin=5,
    charge=0,
    verbose=4,
)

# Define active space
mol.symmetry = 1
mol.build()
mf = scf.RHF(mol)
mf.kernel()

dm = mf.make_rdm1()
mo_coeff = mf.mo_coeff
occ = mf.mo_occ

fe_d_range = range(18, 30)
fe_s_range = range(30, 32)
n_2pz_range = range(3, 6)

fe_d_mo = mo_coeff[:, fe_d_range]
fe_s_mo = mo_coeff[:, fe_s_range]
n_2pz_mo = mo_coeff[:, n_2pz_range]

fe_d_occ = occ[fe_d_range]
fe_s_occ = occ[fe_s_range]
n_2pz_occ = occ[n_2pz_range]

fe_d_mo = fe_d_mo / pyscf.lib.numpy.sqrt(2)
fe_s_mo = fe_s_mo / pyscf.lib.numpy.sqrt(2)
n_2pz_mo = n_2pz_mo / pyscf.lib.numpy.sqrt(2)

fe_d_mo[:, ::2] *= -1
fe_d_mo[:, 1::2] *= -1

active_space_mo = pyscf.lib.misc.block_select(mo_coeff, (fe_d_range + fe_s_range + n_2pz_range))
active_space_occ = pyscf.lib.misc.block_select(occ, (fe_d_range + fe_s_range + n_2pz_range))

# DMRG-CASSCF
dmrgscf.addons.symmetry_adapted_active_space(mol, active_space_mo, active_space_occ)
mc = mcscf.CASCI(mf, active_space_mo, active_space_occ)

# Calculate triplet energy
mc.kernel(state_averaged=False, root=0, ecore=mf.energy_nuc())
scf_triplet = mc.e_tot

# Calculate quintet energy
mc.kernel(state_averaged=False, root=4, ecore=mf.energy_nuc())
scf_quintet = mc.e_tot

# DMRG-NEVPT2
nevpt2 = nesther.NEVPT2(mf, mo_coeff, dm)
nevpt2.kernel(dmrgscf.casci.cas_natorb(mc), ci=mc.ci, verbose=4)

# Calculate triplet energy
tei = nevpt2.tei
h1e = nevpt2.h1e
e_triplet_nevpt2 = nevpt2.energy(tei, h1e, dmrgscf.casci.cas_natorb(mc), root=0)

# Calculate quintet energy
e_quintet_nevpt2 = nevpt2.energy(tei, h1e, dmrgscf.casci.cas_natorb(mc), root=4)

# Output active space orbitals to molden format
tools.molden_writer(mol, mo_coeff, 'active_space.molden')

print("Triplet energy gap (SCF): ", scf_quintet - scf_triplet)
print("Triplet energy gap (NEVPT2): ", e_quintet_nevpt2 - e_triplet_nevpt2)
```