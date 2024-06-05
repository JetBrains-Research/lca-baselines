 ```python
import numpy as np
import matplotlib.pyplot as plt
from simpeg.meshing.simple_mesh import SimpleMesh
from simpeg.maps import ExpMap, SurfaceMap
from simpeg.utils import sigma2m
from simpeg.electromagnetics.static.dd import DipoleDipole
from simpeg.inversion import BaseInversion
from simpeg.optimization import TrustRegionMinimizer
from simpeg.regulation import Regularization
from simpeg.visualization import plotMap

# Create the model - 2 spheres, one conductive and one resistive
radius = np.r_[0.2, 0.4]
sigma = np.r_[1.0e-2, 1.0e1]
center = np.r_[[0., 0., -0.5], [0., 0., 0.5]]

# Background conductivity
sigma_bkg = 1.0

# Create the mesh
h = np.r_[0.05, 0.05, 0.05]
mesh = SimpleMesh([np.ones(n*3) * h[i] for i, n in enumerate(np.r_[30, 30, 30])])

# Create the survey - Dipole-Dipole array
srcList = [DipoleDipole(
    locs=mesh.vectorCCx.T,
    locs1=mesh.vectorCCx[0, :] + 0.05, locs2=mesh.vectorCCx[0, :] - 0.05,
    locs3=mesh.vectorCCz.T,
    locs4=mesh.vectorCCz.T,
    recLocs=mesh.vectorCCx.T,
    recLocs1=mesh.vectorCCx.T, recLocs2=mesh.vectorCCx.T,
    recLocs3=mesh.vectorCCz.T, recLocs4=mesh.vectorCCz.T,
    srcType='dipole',
    max_distance_x=0.95, max_distance_z=0.95,
    receiver_spacing=0.05,
    transmitter_spacing=0.05,
    store_projected_src=True)]
survey = DipoleDipole(mesh, srcList)

# Create the mapping - Active cells and exponential mapping
actvMap = SurfaceMap(mesh)
expMap = ExpMap(sigma_bkg)
mapping = actvMap * expMap

# Create the problem
problem = survey.makeSyntheticData(
    sigma=mapping*sigma,
    relative_error=0.05,
    noise_realization=0,
    add_noise=False)

# Create the inversion
reg = Regularization()
reg.cell_weights = actvMap.cell_weights
reg.m = sigma
reg.b = sigma_bkg
reg.alpha_s = 1.0
reg.alpha_x = 1.0
reg.alpha_y = 1.0
reg.alpha_z = 1.0

inv = BaseInversion()
inv.formulation = 'l2l2'
inv.regulation_type = 'Identity'
inv.regulation_parameter = reg
inv.inverse_operator = problem.inverse_operator
inv.L2_gradient_weighting = 1.0
inv.trust_region = TrustRegionMinimizer()

# Run the inversion
inv_data = inv.run(problem.dpred, np.log(sigma))

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plotMap(axs[0], inv_data.m, vtype='log', cmap='viridis')
plotMap(axs[1], sigma, vtype='log', cmap='viridis')
plt.show()
```