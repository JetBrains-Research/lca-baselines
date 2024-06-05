  import numpy as np
import matplotlib.pyplot as plt
from SimPEG import Mesh, Maps, SolverLU
from SimPEG.EM import DC
from SimPEG.Utils import mkvc

# Define the mesh
hx = 5.
hy = 5.
hz = 5.
npad = 10
cs = 10.
mesh = Mesh.CylMesh([hx, hy, hz], [cs, cs, cs], npad, '00C')

# Define the model
n_sphere1 = 10
n_sphere2 = 10
sphere1_radius = 5.
sphere2_radius = 10.
background_conductivity = 1e-8
conductivity_sphere1 = 1e-4
conductivity_sphere2 = 1e-2

# Define the survey
n_dipoles = 10
dipole_radius = 0.5
dipole_azimuth = np.linspace(0, 360, n_dipoles, endpoint=False)
dipole_dipole_ratio = 1.

# Define the Tikhonov regularization
alpha = 1e-5

# Create the synthetic survey
survey = DC.Survey()
for i in range(n_dipoles):
    survey.add_dipole_dipole(
        [0., 0., 0.],
        [dipole_radius*np.cos(np.deg2rad(dipole_azimuth[i])),
         dipole_radius*np.sin(np.deg2rad(dipole_azimuth[i])),
         0.],
        [dipole_dipole_ratio*dipole_radius*np.cos(np.deg2rad(dipole_azimuth[i])),
         dipole_dipole_ratio*dipole_radius*np.sin(np.deg2rad(dipole_azimuth[i])),
         0.],
        np.log(1e-8), np.log(1e-8)
    )

# Create the active cells mapping
active_cells = np.zeros(mesh.nC, dtype=bool)
for i in range(n_sphere1):
    active_cells = np.logical_or(active_cells,
                                 mesh.gridCC[:, 0]**2 + mesh.gridCC[:, 1]**2 < sphere1_radius**2)
for i in range(n_sphere2):
    active_cells = np.logical_or(active_cells,
                                 mesh.gridCC[:, 0]**2 + mesh.gridCC[:, 1]**2 < sphere2_radius**2)
active_map = Maps.InjectActiveCells(mesh, active_cells, np.log(background_conductivity))

# Create the exponential mapping
exp_map = Maps.ExpMap(mesh)

# Create the Tikhonov regularization
reg = DC.Tikhonov(mesh, alpha=alpha)

# Create the DC problem
prob = DC.Problem3D_CC(mesh, sigmaMap=exp_map*active_map)
prob.Solver = SolverLU

# Assign the model and data
prob.model = np.ones(mesh.nC) * background_conductivity
prob.data_pair = survey

# Run the inversion
mrec = prob.make_synthetic_data(prob.model, data_pair=survey, relative_error=0.01)
m = prob.fit(mrec, reg)

# Plot the results
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(mesh.gridCC[:, 0], mesh.gridCC[:, 1], 'k.', ms=0.5)
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')
axs[0, 0].set_title('Ground truth')
axs[0, 0].axis('equal')
axs[0, 1].plot(mesh.gridCC[:, 0], mkvc(m), 'k.', ms=0.5)
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('z')
axs[0, 1].set_title('Inverted model')
axs[0, 1].axis('equal')
axs[1, 0].plot(mesh.gridCC[:, 0], mesh.gridCC[:, 2], 'k.', ms=0.5)
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('y')
axs[1, 0].set_title('Ground truth')
axs[1, 0].axis('equal')
axs[1, 1].plot(mesh.gridCC[:, 0], mkvc(m), 'k.', ms=0.5)
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('z')
axs[1, 1].set_title('Inverted model')
axs[1, 1].axis('equal')
plt.tight_layout()
plt.show()