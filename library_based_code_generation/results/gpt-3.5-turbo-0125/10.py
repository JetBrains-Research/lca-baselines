from SimPEG import Mesh, Maps, Utils, EM
import numpy as np

# Set up cylindrically symmetric mesh
cs = 10.
ncz = 20
npad = 10
hx = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
mesh = Mesh.CylMesh([hx, 1, hx], '00C')

# Define geologic parameters
sigma = 1e-2
chi = 0.1
m = np.ones(mesh.nC) * chi

# Define mappings
mapping = Maps.IdentityMap(mesh)

# Set up FDEM problem and survey
rxOffset = 7.86
rx = EM.FDEM.Rx.Point_dbdt(np.array([[0., 0., 0.]]), 'z')
src = EM.FDEM.Src.MagDipole([rx], freq=1000.)
survey = EM.FDEM.Survey([src])
problem = EM.FDEM.Problem3D_b(mesh, sigma=sigma, muMap=mapping, chiMap=mapping)
problem.pair(survey)

# Perform FDEM inversion
dmis = EM.FDEM.DataMisfit.l2_DataMisfit(survey)
reg = EM.Tikhonov(mesh, alpha_s=1., alpha_x=1.)
opt = EM.Optimization.InexactGaussNewton(maxIter=10)
invProb = EM.InvProblem.BaseInvProblem(dmis, reg, opt)
inv = EM.Inversion.BaseInversion(invProb)
m0 = np.ones(mesh.nC) * chi
mrec = inv.run(m0)

# Set up inversion directives and run inversion
directiveList = [EM.Directives.BetaSchedule(), EM.Directives.TargetMisfit()]
inv.directiveList = directiveList
mrec = inv.run(m0)

# Plot conductivity model, permeability model, and data misfits
if flag:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    mesh.plotSlice(np.log10(sigma), ax=ax[0])
    mesh.plotSlice(m, ax=ax[1])
    survey.dobs = survey.dpred(mrec)
    survey.plotResiduals(ax=ax[2])