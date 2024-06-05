 import numpy as np
from pyscf import gto, scf, ci, ao2mo

def compute_coulomb(mol1, mol2, sh1, sh2):
    au2ev = 27.21138602
    charge1 = mol1.atom_charges()
    charge2 = mol2.atom_charges()
    coords1 = mol1.coords()
    coords2 = mol2.coords()
    coulomb = 0
    for i in range(len(coords1)):
        for j in range(len(coords2)):
            dist = np.linalg.norm(coords1[i] - coords2[j])
            coulomb += charge1[i] * charge2[j] / dist
    return coulomb * au2ev

def compute_exchange(mol1, mol2, sh1, sh2):
    au2ev = 27.21138602
    s = np.dot(sh1, sh2)
    exchange = -au2ev * np.abs(s)**2
    return exchange

def compute_jk(mol1, mol2, c1, c2, eri1, eri2):
    sh1 = mol1.intor("int1e_ovlp")
    sh2 = mol2.intor("int1e_ovlp")
    eri1 = ao2mo.restore(eri1, mol1.nao_nr(), mol1.nao_nr())
    eri2 = ao2mo.restore(eri2, mol2.nao_nr(), mol2.nao_nr())
    j = np.einsum("ij,kl,ijab->klab", c1, eri1, c1)
    j += np.einsum("ij,kl,ijab->klab", c2, eri2, c2)
    k = np.einsum("ij,ab,ijab->klab", c1, eri1, c1)
    k += np.einsum("ij,ab,ijab->klab", c2, eri2, c2)
    jk = j + k
    return jk

def compute_dft_xc(mol1, mol2, c1, c2, dm1, dm2):
    xc = scf.xcfunc.get_xcfun("b3lyp")
    dm12 = np.zeros((mol1.nao_nr(), mol2.nao_nr()))
    dm21 = np.zeros((mol2.nao_nr(), mol1.nao_nr()))
    dm12 = np.dot(c1, np.dot(dm1, c1.T))
    dm21 = np.dot(c2, np.dot(dm2, c2.T))
    dm12 = dm12.reshape(mol1.nao_nr(), -1)
    dm21 = dm21.reshape(-1, mol2.nao_nr())
    dm12 = np.dot(sh2, dm12)
    dm21 = np.dot(dm21, sh1)
    dm12 = np.dot(dm12, dm21)
    dft_xc = xc.exc(mol1, dm1 + dm12, dm2 + dm21)[0]
    return dft_xc

def compute_coupling(mol1, mol2, state1, state2):
    eri1 = ci.CIS(mol1, state1).kernel()
    eri2 = ci.CIS(mol2, state2).kernel()
    c1 = ci.CIS(mol1, state1).ci
    c2 = ci.CIS(mol2, state2).ci
    sh1 = mol1.intor("int1e_ovlp")
    sh2 = mol2.intor("int1e_ovlp")
    eri1 = ao2mo.restore(eri1, mol1.nao_nr(), mol1.nao_nr())
    eri2 = ao2mo.restore(eri2, mol2.nao_nr(), mol2.nao_nr())
    jk = compute_jk(mol1, mol2, c1, c2, eri1, eri2)
    dft_xc = compute_dft_xc(mol1, mol2, c1, c2, mol1.make_rdm1(c1), mol2.make_rdm1(c2))
    coulomb = compute_coulomb(mol1, mol2, sh1, sh2)
    exchange = compute_exchange(mol1, mol2, sh1, sh2)
    coupling = jk + coulomb + exchange + dft_xc
    return coupling

mol1 = gto.M(atom="H 0 0 0; H 0 0 1.5", basis="6-31g")
mol2 = gto.M(atom="H 3 0 0; H 3 0 1.5", basis="6-31g")
state1 = 1
state2 = 1
coupling = compute_coupling(mol1, mol2, state1, state2)
print(coupling)