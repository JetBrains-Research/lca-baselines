  from pyscf import gto, scf, cc

# Build the cell
mol = gto.M(
    atom='C 0.75 0.75 0.75; C 1.25 0.75 0.75',
    basis='6-31g',
    pseudo='gth-pade',
    a=2.0,
    unit='B',
    kpt=(2, 2, 2)
)

# Perform KHF and KCCSD calculations with 2x2x2 k-points
khf = scf.RHF(mol)
khf.kernel()
kccsd = cc.CCSD(khf)
kccsd.kernel()
print('Total energy per unit cell (KHF+KCCSD):', kccsd.e_tot)

# Perform KHF and KCCSD calculations for a single k-point
khf = scf.RHF(mol)
khf.kernel()
kccsd = cc.CCSD(khf)
kccsd.kernel()
print('Total energy per unit cell (KHF+KCCSD):', kccsd.e_tot)

# Perform a single k-point calculation using the RHF method, run RCCSD, and print the total energy per unit cell
rhf = scf.RHF(mol)
rhf.kernel()
rccsd = cc.CCSD(rhf)
rccsd.kernel()
print('Total energy per unit cell (RHF+RCCSD):', rccsd.e_tot)

# Calculate and print the RCCSD energy based on CCSD density matrices
rccsd_dm = rccsd.make_rdm1()
rccsd_energy = rccsd.ccsd_energy(rccsd_dm)
print('RCCSD energy:', rccsd_energy)

# Convert the RHF object to a UHF object, run UCCSD, and print the total energy per unit cell at the k-point
uhf = scf.UHF(mol)
uhf.kernel()
uccsd = cc.CCSD(uhf)
uccsd.kernel()
print('Total energy per unit cell (UHF+UCCSD):', uccsd.e_tot)

# Calculate and print the UCCSD energy based on CCSD density matrices
uccsd_dm = uccsd.make_rdm1()
uccsd_energy = uccsd.ccsd_energy(uccsd_dm)
print('UCCSD energy:', uccsd_energy)

# Convert the UHF object to a GHF object, run GCCSD, and print the total energy per unit cell at the k-point
ghf = scf.GHF(mol)
ghf.kernel()
gccsd = cc.CCSD(ghf)
gccsd.kernel()
print('Total energy per unit cell (GHF+GCCSD):', gccsd.e_tot)

# Calculate and print the GCCSD energy based on CCSD density matrices
gccsd_dm = gccsd.make_rdm1()
gccsd_energy = gccsd.ccsd_energy(gccsd_dm)
print('GCCSD energy:', gccsd_energy)