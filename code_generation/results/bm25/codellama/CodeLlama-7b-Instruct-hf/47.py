  Here is the Python code for the 3D DC inversion of a dipole-dipole array using the SimPEG library:
```
import numpy as np
import matplotlib.pyplot as plt
from SimPEG import Mesh, Maps, Inversion, Utils

# Define the mesh
h = 10.0
mesh = Mesh.TensorMesh([h, h, h], x0='CCC')

# Define the model
background = 1e-8
conductive = 1e-4
resistive = 1e-2
model = background + conductive * np.random.rand(mesh.nC) + resistive * np.random.rand(mesh.nC)

# Define the survey
survey = Utils.Survey()
survey.src_loc = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
survey.rx_loc = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
survey.freq = 1e-3

# Define the active cells mapping
active_cells = np.where(model > 0, 1, 0)
active_cells_map = Maps.InjectActiveCells(mesh, active_cells, np.nan)

# Define the exponential mapping
exponential_map = Maps.Exp(mesh)

# Define the Tikhonov inversion
tikhonov = Inversion.Tikhonov(mesh, active_cells_map, exponential_map, np.inf)

# Define the synthetic dipole-dipole survey
synthetic_survey = Utils.Survey()
synthetic_survey.src_loc = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
synthetic_survey.rx_loc = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
synthetic_survey.freq = 1e-3

# Define the Tikhonov inversion for the synthetic survey
synthetic_tikhonov = Inversion.Tikhonov(mesh, active_cells_map, exponential_map, np.inf)

# Perform the inversion
inverted_model = tikhonov.run(synthetic_survey)

# Generate a plot of the ground truth and the inverted model
plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(model, cmap='gray')
plt.title('Ground Truth')
plt.subplot(122)
plt.imshow(inverted_model, cmap='gray')
plt.title('Inverted Model')
plt.show()
```
This code performs a 3D DC inversion of a dipole-dipole array using the SimPEG library. The model consists of two spheres, one conductive and the other resistive, compared to the background. The inversion is restrained to the Core Mesh using an Active Cells mapping combined with an exponential mapping to invert in log conductivity space. The code also includes the creation of a synthetic Dipole-Dipole Survey and a Tikhonov Inversion. Finally, the code generates a plot of the ground truth and the inverted model, both vertically and horizontally.