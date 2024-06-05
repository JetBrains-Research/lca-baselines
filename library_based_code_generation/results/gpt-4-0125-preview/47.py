```python
import numpy as np
import matplotlib.pyplot as plt
from SimPEG import (
    Mesh, Maps, Utils, EM, Survey, DataMisfit, Regularization,
    Optimization, Inversion, InvProblem, Directives
)

# Create a mesh
cs = 25.0
npad = 11
hx = [(cs, npad, -1.3), (cs, 41), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, 41), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, 20)]
mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')

# Model setup: two spheres in a homogeneous background
sigma_background = 1e-2  # Background conductivity
sigma_conductor = 1e-1   # Conductive sphere
sigma_resistor = 1e-3    # Resistive sphere
sphere_conductor = [np.r_[320., 320., -200.], 80.]  # center and radius
sphere_resistor = [np.r_[480., 480., -200.], 80.]   # center and radius

# Create model
model = sigma_background * np.ones(mesh.nC)
inds_conductor = Utils.ModelBuilder.getIndicesSphere(sphere_conductor[0], sphere_conductor[1], mesh.gridCC)
inds_resistor = Utils.ModelBuilder.getIndicesSphere(sphere_resistor[0], sphere_resistor[1], mesh.gridCC)
model[inds_conductor] = sigma_conductor
model[inds_resistor] = sigma_resistor

# Mapping and active cells
actv = mesh.gridCC[:, 2] < 0  # Only consider subsurface
mapping = Maps.ExpMap(mesh) * Maps.InjectActiveCells(mesh, actv, np.log(sigma_background), nC=mesh.nC)

# Survey setup
srcList = []
nSrc = 10  # Number of sources
for i in range(nSrc):
    locA = np.r_[80 + i*40, 320., -340.]
    locB = np.r_[80 + i*40, 480., -340.]
    locM = np.r_[120 + i*40, 320., -340.]
    locN = np.r_[120 + i*40, 480., -340.]
    src = EM.Static.DC.Src.Dipole([locM, locN], locA, locB)
    srcList.append(src)
survey = EM.Static.DC.Survey(srcList)

# Problem and data
problem = EM.Static.DC.Problem3D_CC(mesh, sigmaMap=mapping)
problem.pair(survey)
data = problem.makeSyntheticData(model, relative_error=0.05, add_noise=True)

# Inversion
dmis = DataMisfit.l2_DataMisfit(survey)
reg = Regularization.Simple(mesh, indActive=actv, mapping=Maps.IdentityMap(nP=int(np.sum(actv))))
opt = Optimization.InexactGaussNewton(maxIter=10)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
directives = [Directives.BetaSchedule(coolingFactor=5, coolingRate=2),
              Directives.TargetMisfit()]
inv = Inversion.BaseInversion(invProb, directiveList=directives)

# Run inversion
m0 = np.log(np.ones(actv.sum())*sigma_background)
mrec = inv.run(m0)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Ground truth
model_plot = np.log10(model[actv]).reshape(mesh.vnC[2], mesh.vnC[1], mesh.vnC[0])[::-1, :, :]
axs[0].imshow(model_plot[:, :, int(mesh.vnC[0]/2)], extent=[mesh.vectorNy[0], mesh.vectorNy[-1], mesh.vectorNz[0], mesh.vectorNz[-1]])
axs[0].set_title('True Model')

# Inverted model
recovered_model = mapping*mrec
recovered_plot = np.log10(recovered_model).reshape(mesh.vnC[2], mesh.vnC[1], mesh.vnC[0])[::-1, :, :]
axs[1].imshow(recovered_plot[:, :, int(mesh.vnC[0]/2)], extent=[mesh.vectorNy[0], mesh.vectorNy[-1], mesh.vectorNz[0], mesh.vectorNz[-1]])
axs[1].set_title('Inverted Model')

plt.tight_layout()
plt.show()
```