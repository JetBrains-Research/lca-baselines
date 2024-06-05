```python
import numpy as np
import matplotlib.pyplot as plt
from SimPEG import Mesh, Maps, Utils, EM, Survey, DataMisfit, Regularization, Optimization, Inversion, Directives, InvProblem
try:
    from pymatsolver import PardisoSolver as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

# Flag for plotting
plot_flag = True

# Set up cylindrically symmetric mesh
cs, ncx, ncz, npad = 10., 20, 20, 15
hx = [(cs, ncx), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
mesh = Mesh.CylMesh([hx, 1, hz], '00C')

# Geologic parameters and electrical conductivity
sigma_background = 1e-3  # S/m
layer_conductivity = 1e-2  # S/m
layer_thickness = 50  # meters

# Define layer
layer_top = mesh.vectorCCz[np.argmin(np.abs(mesh.vectorCCz - (-layer_thickness)))]
layer_bottom = -layer_thickness * 2
layer_inds = np.logical_and(mesh.vectorCCz <= layer_top, mesh.vectorCCz >= layer_bottom)
sigma = np.ones(mesh.nCz) * sigma_background
sigma[layer_inds] = layer_conductivity

# Set up the relative magnetic permeability
mu = np.ones(mesh.nCz)

# Define mappings
sigmaMap = Maps.InjectActiveCells(mesh, layer_inds, sigma_background)
muMap = Maps.InjectActiveCells(mesh, layer_inds, 1.0)

# Set up FDEM problem and survey
frequency = np.logspace(1, 3, 20)
rx = EM.FDEM.Rx.Point_bSecondary(loc=np.array([[0., 0., -layer_thickness/2]]), orientation='z', component='real')
src = EM.FDEM.Src.MagDipole([rx], freq=frequency, loc=np.array([0., 0., 0.]))
survey = EM.FDEM.Survey([src])
problem = EM.FDEM.Problem3D_b(mesh, sigmaMap=sigmaMap, muMap=muMap)
problem.Solver = Solver
problem.pair(survey)

# Generate synthetic data
data = survey.dpred(np.r_[sigma[layer_inds], mu[layer_inds]])

# Set up data misfit, regularization, and inversion
dmis = DataMisfit.l2_DataMisfit(survey)
reg = Regularization.Simple(mesh, indActive=layer_inds)
opt = Optimization.InexactGaussNewton(maxIter=10)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
directives = [Directives.BetaSchedule(coolingFactor=5, coolingRate=2),
              Directives.TargetMisfit()]
inv = Inversion.BaseInversion(invProb, directiveList=directives)

# Run inversion
recovered_model = inv.run(data)

# Plotting
if plot_flag:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(mesh.vectorCCz, sigma, label='True Conductivity')
    axs[0].plot(mesh.vectorCCz[layer_inds], recovered_model[:np.sum(layer_inds)], 'k--', label='Recovered Conductivity')
    axs[0].legend()
    axs[0].set_title('Conductivity Model')
    
    axs[1].plot(mesh.vectorCCz, mu, label='True Permeability')
    axs[1].plot(mesh.vectorCCz[layer_inds], recovered_model[np.sum(layer_inds):], 'k--', label='Recovered Permeability')
    axs[1].legend()
    axs[1].set_title('Permeability Model')
    
    axs[2].plot(frequency, data, 'bo-', label='True Data')
    axs[2].plot(frequency, survey.dpred(recovered_model), 'kx-', label='Predicted Data')
    axs[2].legend()
    axs[2].set_title('Data Misfit')
    axs[2].set_xscale('log')
    
    plt.tight_layout()
    plt.show()
```