 ```python
import numpy as np
import matplotlib.pyplot as plt
from simpeg.meshing.simple_mesh import SimpleMesh
from simpeg.maps import ExpMap, SurfaceMap
from simpeg.survey import Survey
from simpeg.data import Data
from simpeg.inversion import BaseInversion
from simpeg.optimization import TrustRegionMinimizer
from simpeg.regulation import Regularization
from simpeg.utils import plot2Ddata
from simpeg.electromagnetics.static.resistivity import DC, DipoleDipole
from simpeg.boundary import Hankel

# Create the model
csphere = [(0.2, 1.), (0.4, 0.1)]
rsphere = [(0.3, 10.), (0.5, 0.1)]
model = np.ones((1, 1, 1)) * 1.0
for r, s in csphere:
    model[0, r*10:-r*10, r*10:-r*10] = s
for r, s in rsphere:
    model[0, r*10:-r*10, r*10:-r*10] = s

# Create the mesh
h = np.ones(10) * 0.05
mesh = SimpleMesh([h, h, h], '0D', '0D', '0D')

# Create the survey
srcList = [DipoleDipole(5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, Ab = 1.)]
survey = Survey(srcList)

# Create the problem
problem = DC.Problem3D_CC(mesh)
problem.pair(survey)
problem.set_eps(1.)

# Create the survey data
sigma = model.ravel()
problem. survey.dpred(sigma)

# Create the inversion
reg = Regularization()
reg.cell_weights = mesh.cell_volumes
reg.make()
reg.JtJ = reg.JtJ + reg.alpha*np.eye(reg.JtJ.shape[0])

inversion = BaseInversion()
inversion.regulation_type = 'Survey'
inversion.regulation_parameter = reg
inversion.inverse_method = TrustRegionMinimizer
inversion.trust_region_strategy = 'Dogleg'
inversion.trust_region_initial_guess = 1.
inversion.trust_region_growth_factor = 2.
inversion.trust_region_step_length = 0.5
inversion.max_iterations = 100
inversion.tol = 1e-3
inversion.lower = 1e-6
inversion.upper = 1e6
inversion.beta = 1.
inversion.verbose = True

# Run the inversion
m0 = np.ones(mesh.nC) * 1.
inversion.run(problem, m0)

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plot2Ddata(axs[0], mesh, model.T, 'Model', 'Vertical Cross Section', cmap='jet')
plot2Ddata(axs[1], mesh, inversion.m_opt.T, 'Inverted Model', 'Vertical Cross Section', cmap='jet')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plot2Ddata(axs[0], mesh, model.T, 'Model', 'Horizontal Cross Section', cmap='jet')
plot2Ddata(axs[1], mesh, inversion.m_opt.T, 'Inverted Model', 'Horizontal Cross Section', cmap='jet')
plt.show()
```