import numpy as np
import matplotlib.pyplot as plt
import discretize
from SimPEG import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    utils,
)

# Create mesh
cs = 5.0
hx = [(cs, 5, -1.3), (cs, 40), (cs, 5, 1.3)]
hy = [(cs, 5, -1.3), (cs, 40), (cs, 5, 1.3)]
hz = [(cs, 5, -1.3), (cs, 20)]
mesh = discretize.TensorMesh([hx, hy, hz], "CCN")

# Create model
model = np.ones(mesh.nC) * 1e-2
ind_conductive = utils.model_builder.getIndicesSphere(np.r_[-100, 0, -50], 20, mesh.gridCC)
ind_resistive = utils.model_builder.getIndicesSphere(np.r_[100, 0, -50], 20, mesh.gridCC)
model[ind_conductive] = 1e-1
model[ind_resistive] = 1e-3

# Create mapping
actv = np.ones(mesh.nC, dtype=bool)
mapping = maps.InjectActiveCells(mesh, actv, np.log(1e-8), nC=mesh.nC)
mapping = maps.ExpMap(mesh) * mapping

# Create survey
rxloc = np.array([[0, 0, 0], [20, 0, 0]])
srcloc = np.array([[0, 0, 0], [40, 0, 0]])
rx = data.Data(survey, rxloc)
src = data.Data(survey, srcloc)
survey = data.survey.SurveyDC(src, rx)

# Create problem
problem = data_misfit.Simulation3DCellCentered(
    mesh, survey=survey, sigmaMap=mapping, Solver=Solver
)

# Create regularization
reg = regularization.Tikhonov(mesh)

# Create optimization
opt = optimization.InexactGaussNewton(maxIter=20)

# Create inversion
invProb = inverse_problem.BaseInvProblem(problem, reg, opt)

# Create directives
beta = directives.BetaSchedule()
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
target = directives.TargetMisfit()
update_sense = directives.UpdateSensitivityWeights()
update_Jacobi = directives.UpdatePreconditioner()

# Run inversion
inv = inversion.BaseInversion(invProb, directiveList=[beta, betaest, target, update_sense, update_Jacobi])
m0 = np.ones(mesh.nC) * np.log(1e-8)
mrec = inv.run(m0)

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
mesh.plotSlice(np.log10(model), ax=ax[0], normal="Y", ind=10, clim=(-4, -1), grid=True, pcolorOpts={"cmap": "viridis"})
mesh.plotSlice(mrec, ax=ax[1], normal="Y", ind=10, clim=(-4, -1), grid=True, pcolorOpts={"cmap": "viridis"})
plt.show()