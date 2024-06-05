```python
import numpy as np
import matplotlib.pyplot as plt
from SimPEG import (
    Mesh, Maps, Utils, EM, Survey, DataMisfit, Regularization,
    Optimization, Inversion, InvProblem, Directives
)

# Create a mesh
cs = 25.0
npad = 10
hx = [(cs, npad, -1.3), (cs, 100), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, 100), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, 60), (cs, npad, 1.3)]
mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')

# Model setup: two spheres in a homogeneous background
sigma_background = 1e-2  # Background conductivity
sigma_sphere1 = 1e-1  # Conductive sphere
sigma_sphere2 = 1e-3  # Resistive sphere
sphere1_center = np.r_[50., 50., -100.]
sphere2_center = np.r_[-50., -50., -150.]
sphere_radius = 30.

# Create model
model = sigma_background * np.ones(mesh.nC)
sphere1 = (mesh.gridCC - sphere1_center)**2
sphere1 = np.sqrt(sphere1[:, 0] + sphere1[:, 1] + sphere1[:, 2]) < sphere_radius
model[sphere1] = sigma_sphere1
sphere2 = (mesh.gridCC - sphere2_center)**2
sphere2 = np.sqrt(sphere2[:, 0] + sphere2[:, 1] + sphere2[:, 2]) < sphere_radius
model[sphere2] = sigma_sphere2

# Mapping and active cells
actv = Utils.ModelBuilder.getIndicesSphere(np.r_[0., 0., -75.], 200., mesh.gridCC)
actMap = Maps.InjectActiveCells(mesh, actv, np.log(sigma_background))
mapping = Maps.ExpMap(mesh) * actMap

# Survey setup
srcList = []
n = 10
for i in range(n):
    for j in range(i+1, n):
        locA = np.r_[i*50., 0., 0.]
        locB = np.r_[j*50., 0., 0.]
        locM = np.r_[(i+0.5)*50., 0., 0.]
        locN = np.r_[(j-0.5)*50., 0., 0.]
        rx = EM.Static.Dipole(locsM=locM, locsN=locN)
        src = EM.Static.Dipole([rx], locA=locA, locB=locB)
        srcList.append(src)
survey = Survey(srcList)
problem = EM.Static.SIP.Problem3D_CC(mesh, sigmaMap=mapping)
problem.pair(survey)

# Generate synthetic data
data = problem.makeSyntheticData(model, noise_floor=1e-3, add_noise=True)

# Inversion
dmisfit = DataMisfit.l2_DataMisfit(survey)
reg = Regularization.Simple(mesh, indActive=actv, mapping=Maps.IdentityMap(nP=int(np.sum(actv))))
opt = Optimization.InexactGaussNewton(maxIter=10)
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
directives = [
    Directives.BetaSchedule(coolingFactor=5, coolingRate=2),
    Directives.TargetMisfit()
]
inv = Inversion.BaseInversion(invProb, directiveList=directives)

# Run inversion
recovered_model = inv.run(np.log(model[actv]))

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Ground truth
model_plot = np.log10(model).reshape(mesh.vnC, order="F")
axs[0].imshow(model_plot[:, :, int(mesh.nCz/2)], origin="lower")
axs[0].set_title("Ground Truth (Vertical Section)")

# Inverted model
recovered_model_plot = np.log10(np.exp(recovered_model)).reshape((np.sum(actv),), order="F")
recovered_model_full = np.log10(sigma_background)*np.ones(mesh.nC)
recovered_model_full[actv] = recovered_model_plot
recovered_model_full = recovered_model_full.reshape(mesh.vnC, order="F")
axs[1].imshow(recovered_model_full[:, :, int(mesh.nCz/2)], origin="lower")
axs[1].set_title("Inverted Model (Vertical Section)")

plt.tight_layout()
plt.show()
```