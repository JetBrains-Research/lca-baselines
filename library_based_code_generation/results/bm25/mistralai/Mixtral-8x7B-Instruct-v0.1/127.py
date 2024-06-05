from pyscf import gto, scf, ci, ao2mo
import numpy as np

def calculate\_cis(mol):
mol.spin = 0
mf = scf.RHF(mol)
mf.kernel()

cis = ci.CIS(mf)
cis.kernel()

return cis

def intermolecular\_integrals(mol1, mol2):
intor\_ao = ao2mo.general\_integrals(mol1, mol2, aosym='s2', compact=False)

return intor\_ao

def transform\_integrals\_mo(intor\_ao, cis1, cis2):
mo1 = cis1.mo\_coeff
mo2 = cis2.mo\_coeff

intor\_mo = ao2mo.restore(1, intor\_ao, (mo1, mo2))

return intor\_mo

def coulomb\_integrals(intor\_mo):
J = ao2mo.general\_integrals(intor\_mo, compact=False, aosym='s4', outtype=np.complex128)
J = np.einsum('ijab,ijab->', J, np.ones((intor\_mo.shape[0], intor\_mo.shape[0])))

return J

def exchange\_integrals(intor\_mo):
K = ao2mo.exchange(intor\_mo)
K = np.einsum('ijab,ijab->', K, np.ones((intor\_mo.shape[0], intor\_mo.shape[0])))

return K

def coupling\_term(cis1, cis2, intor\_mo, J, K):
n\_occ1, n\_vir1 = cis1.nmo\_occ, cis1.nmo\_vir
n\_occ2, n\_vir2 = cis2.nmo\_occ, cis2.nmo\_vir

mo1\_occ = cis1.mo\_coeff[:, :n\_occ1]
mo1\_vir = cis1.mo\_coeff[:, n\_occ1:]
mo2\_occ = cis2.mo\_coeff[:, :n\_occ2]
mo2\_vir = cis2.mo\_coeff[:, n\_occ2:]

J\_occ = J[:n\_occ1, :n\_occ1]
J\_vir = J[n\_occ1:, n\_occ1:]
J\_ov = J[:n\_occ1, n\_occ1:]
J\_vo = J[n\_occ1:, :n\_occ1]

K\_occ = K[:n\_occ1, :n\_occ1]
K\_vir = K[n\_occ1:, n\_occ1:]
K\_ov = K[:n\_occ1, n\_occ1:]
K\_vo = K[n\_occ1:, :n\_occ1]

DFT\_XC = 0.0 # DFT XC contribution

J\_term = np.einsum('ij,kl,ijkl->', mo1\_occ.T, mo2\_occ.T, J\_occ)
K\_term = np.einsum('ij,kl,ijkl->', mo1\_occ.T, mo2\_occ.T, K\_occ)
DFT\_XC\_term = np.einsum('ij,kl,ijkl->', mo1\_occ.T, mo2\_occ.T, DFT\_XC)

J\_term += np.einsum('ij,kl,ijkl->', mo1\_vir.T, mo2\_vir.T, J\_vir)
K\_term += np.einsum('ij,kl,ijkl->', mo1\_vir.T, mo2\_vir.T, K\_vir)
DFT\_XC\_term += np.einsum('ij,kl,ijkl->', mo1\_vir.T, mo2\_vir.T, DFT\_XC)

J\_term += np.einsum('ij,kl,ijkl->', mo1\_vir.T, mo2\_occ.T, J\_ov)
K\_term += np.einsum('ij,kl,ijkl->', mo1\_vir.T, mo2\_occ.T, K\_ov)
DFT\_XC\_term += np.einsum('ij,kl,ijkl->', mo1\_vir.T, mo2\_occ.T, DFT\_XC)

J\_term += np.einsum('ij,kl,ijkl->', mo1\_occ.T, mo2\_vir.T, J\_vo)
K\_term += np.einsum('ij,kl,ijkl->', mo1\_occ.T, mo2\_vir.T, K\_vo)
DFT\_XC\_term += np.einsum('ij,kl,ijkl->', mo1\_occ.T, mo2\_vir.T, DFT\_XC)

coupling = J\_term - K\_term + DFT\_XC\_term

return coupling

mol1 = gto.M(atom='H 0 0 0; H 0 0 1.5', basis='6-31g')
mol2 = gto.M(atom='H 3 0 0; H 3 0 1.5', basis='6-31g')

cis1 = calculate\_cis(mol1)
cis2 = calculate\_cis(mol2)

intor\_ao = intermolecular\_integrals(mol1, mol2)
intor\_mo = transform\_integrals\_mo(intor\_ao, cis1, cis2)
J = coulomb\_integrals(intor\_mo)
K = exchange\_integrals(intor\_mo)

coupling = coupling\_term(cis1, cis2, intor\_mo, J, K)
print(coupling)