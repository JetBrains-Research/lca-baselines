```python
import numpy as np
import discretize
from SimPEG import (
    maps, 
    utils, 
    surveys, 
    data_misfit, 
    regularization, 
    optimization, 
    inversion, 
    inverse_problem, 
    directives
)
from SimPEG.electromagnetics import frequency_domain as fdem
import matplotlib.pyplot as plt

# Check for PardisoSolver
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

# Flag for plotting
plot_flag = True

# Set up the cylindrically symmetric mesh
cs, ncx, ncz, npad = 10., 20, 20, 15
hx = [(cs, ncx), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
mesh = discretize.CylMesh([hx, 1, hz], '00C')

# Define geologic parameters and electrical conductivity
sigma_background = 1e-2  # S/m
sigma_block = 1e-1  # S/m
chi_background = 0.  # Dimensionless
chi_block = 0.1  # Dimensionless

# Create models
sigma = sigma_background * np.ones(mesh.nC)
chi = chi_background * np.ones(mesh.nC)

# Add a block anomaly
block_ind = utils.model_builder.getIndicesBlock(
    [np.r_[0., 0., -150.], np.r_[0., 0., -50.]], mesh.gridCC
)
sigma[block_ind] = sigma_block
chi[block_ind] = chi_block

# Set up the relative magnetic permeability
mu = 1 + chi

# Define mappings
sigma_map = maps.IdentityMap(nP=mesh.nC)
mu_map = maps.IdentityMap(nP=mesh.nC)

# Set up the FDEM problem and survey
rx_locations = np.array([[50., 0., -100.]])
rx = fdem.receivers.PointMagneticFluxDensity(rx_locations, orientation='z', component='real')
src = fdem.sources.MagDipole([rx], frequency=1e3, location=np.array([0., 0., 0.]))
survey = fdem.Survey([src])

# Problem
problem = fdem.Problem3D_h(mesh, sigmaMap=sigma_map, muMap=mu_map, Solver=Solver)
problem.pair(survey)

# Generate synthetic data
data = survey.dpred(np.r_[sigma, mu])

# Set up inversion
dmisfit = data_misfit.L2DataMisfit(data=data, simulation=problem)
reg = regularization.Tikhonov(mesh, mapping=sigma_map)
opt = optimization.InexactGaussNewton(maxIter=10)
inv_prob = inverse_problem.BaseInvProblem(dmisfit, reg, opt)
directives_list = [directives.BetaSchedule(coolingFactor=5, coolingRate=2), directives.TargetMisfit()]
inv = inversion.BaseInversion(inv_prob, directiveList=directives_list)

# Run inversion
recovered_model = inv.run(np.r_[sigma, mu])

# Plotting
if plot_flag:
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    
    # Conductivity model
    mesh.plotSlice(sigma, normal='Y', ax=axs[0], grid=True)
    axs[0].set_title('True Conductivity')
    
    # Permeability model
    mesh.plotSlice(mu, normal='Y', ax=axs[1], grid=True)
    axs[1].set_title('True Permeability')
    
    # Data misfit
    axs[2].plot(data, label='Observed Data')
    axs[2].plot(inv_prob.dpred, label='Predicted Data')
    axs[2].legend()
    axs[2].set_title('Data Misfit')
    
    plt.tight_layout()
    plt.show()
```