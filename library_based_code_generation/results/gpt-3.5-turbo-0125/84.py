```python
from pyscf import gto, scf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df as pbcdf
from pyscf.pbc.tools import pywannier90

cell = pbcgto.Cell()
cell.atom = '''
   H 0. 0. 0.
   H 0. 0. 1.
'''
cell.basis = 'sto-3g'
cell.a = '''
   1.0 0.0 0.0
   0.0 1.0 0.0
   0.0 0.0 2.0
'''
cell.verbose = 5
cell.build()

mf = pbcscf.KRHF(cell, kpts=cell.make_kpts([2, 2, 2]))
mf.xc = 'pbe'
mf.kernel()

kpts = cell.make_kpts([2, 2, 2])
kpts -= kpts[0]
kks = pbcscf.KKS(cell, kpts)
kks.mo_coeff = mf.mo_coeff
kks.mo_occ = mf.mo_occ
kks.mo_energy = mf.mo_energy
kks.kernel()

kks_mf = pywannier90.KKSmf(kks)
kks_mf.save()

kks_mf = pywannier90.KKSmf(cell)
kks_mf.load()

mlwf = pywannier90.W90(kks_mf)
mlwf.export_xsf()

mlwf.export_amn(h5file='amn.h5')
mlwf.export_mmn(h5file='mmn.h5')

pywannier90.run(cell, kks_mf, num_iter=100)

w90 = pywannier90.W90(kks_mf)
w90.interpolate_fock()
w90.interpolate_bands()

scf_bands = mf.get_bands()
w90_bands = w90.get_bands()

print(abs(scf_bands - w90_bands))

w90.plot_band()
```