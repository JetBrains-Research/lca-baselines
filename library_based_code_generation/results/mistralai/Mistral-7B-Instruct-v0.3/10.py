 Here is a simplified Python code using the MNE-PyMAG library for FDEM inversion. Please note that this is a basic example and might need adjustments to fit your specific needs. Also, the flag for plotting and the handling of the PardisoSolver are not included in this example.

```python
import numpy as np
from mne.io import read_raw_fif
from mne_pymag.misc import load_model
from mne_pymag.fdem import FDEM, FDEMInversion
from mne_pymag.inversion import InversionDirectives

# Define geologic parameters and electrical conductivity
r_outer = 1000  # outer radius [m]
r_inner = 500   # inner radius [m]
n_layers = 3   # number of layers
layer_thickness = np.array([100, 200, 700])  # [m]
sigma = np.array([1, 10, 1])  # electrical conductivity [S/m]

# Define relative magnetic permeability
mu_r = np.ones(n_layers)

# Set up cylindrically symmetric mesh
mesh = load_model('cylinder_mesh.npz')

# Define FDEM problem and survey
survey = read_raw_fif('survey.fif')  # replace with your survey data
fdem = FDEM(mesh, survey, mu_r=mu_r, sigma=sigma)

# Define inversion directives
directives = InversionDirectives(max_iter=100, tolerance=1e-6)

# Perform FDEM inversion
inversion = FDEMInversion(fdem, directives)
inversion.run()

# The inversion result can be accessed as inversion.result['sigma']
```

This code assumes that you have a cylindrical mesh file named 'cylinder_mesh.npz' and a survey data file named 'survey.fif'. You will need to adjust the geologic parameters, electrical conductivity, and survey data according to your specific problem.

For handling the absence of the PardisoSolver, you can use the SolverLU as a fallback by setting the solver option in the InversionDirectives:

```python
directives = InversionDirectives(max_iter=100, tolerance=1e-6, solver='LU')
```

For plotting the conductivity model, permeability model, and data misfits, you can use the MNE-PyMAG plotting functions after the inversion is complete. The exact code for this will depend on the specific plots you want to generate.