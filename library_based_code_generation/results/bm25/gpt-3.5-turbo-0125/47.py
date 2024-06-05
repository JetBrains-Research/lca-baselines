from SimPEG import Mesh, Maps, Utils, DataMisfit, Regularization, Optimization, InvProblem, Directives, Inversion, PF
import numpy as np

# Create mesh
cs = 25.
hx = [(cs, 5, -1.3), (cs, 40), (cs, 5, 1.3)]
hy = [(cs, 5, -1.3), (cs, 40), (cs, 5, 1.3)]
hz = [(cs, 5, -1.3), (cs, 20)]
mesh = Mesh.TensorMesh([hx, hy, hz], 'CCN')

# Create model
model = np.ones(mesh.nC) * 1e-2
model[(mesh.gridCC[:,0] < 0) & (np.sqrt((mesh.gridCC[:,1])**2 + (mesh.gridCC[:,2])**2) < 200)] = 1e-1
model[(mesh.gridCC[:,0] > 0) & (np.sqrt((mesh.gridCC[:,1])**2 + (mesh.gridCC[:,2])**2) < 200)] = 1e-3

# Create mapping
actvMap = Maps.InjectActiveCells(mesh, np.ones(mesh.nC) == 1)
mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * actvMap

# Create problem
problem = DC.Problem3D_CC(mesh, sigmaMap=mapping)
problem.Solver = Solver

# Create survey
survey = DC.Survey()
survey.dipole_md = 10
survey.dipole_azimuth = 0
survey.dipole_elevation = 0
survey.srcLocs = np.array([[0, 0, 0], [10, 0, 0]])
survey.recLocs = np.array([[20, 0, 0], [30, 0, 0]])
survey.components = ['z']
survey.std = 0.05
problem.pair(survey)

# Create data
data = survey.dpred(model)

# Create data misfit
dmisfit = DataMisfit.l2_DataMisfit(survey)
dmisfit.W = 1./survey.std

# Create regularization
reg = Regularization.Simple(mesh)
reg.mref = np.zeros(mesh.nC)

# Create optimization
opt = Optimization.InexactGaussNewton(maxIter=10)

# Create inversion
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
inv = Inversion.BaseInversion(invProb)

# Run inversion
m0 = np.ones(mesh.nC) * 1e-2
mrec = inv.run(m0)

# Plot results
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
mesh.plotSlice(np.log10(model), ax=ax[0], normal='Y', ind=10, clim=(-3, -1), grid=True, pcolorOpts={'cmap':'viridis'})
mesh.plotSlice(np.log10(mrec), ax=ax[1], normal='Y', ind=10, clim=(-3, -1), grid=True, pcolorOpts={'cmap':'viridis'})
plt.show()