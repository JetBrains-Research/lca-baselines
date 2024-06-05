  import numpy as np
import matplotlib.pyplot as plt
from SimPEG import Mesh, Maps, SolverLU, SolverPardiso
from SimPEG.EM import FDEM
from SimPEG.Utils import mkvc

# Set up the mesh
cs = 5.0
npad = 10
hx = [(cs, npad, -1.3), (cs, 100), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, 100), (cs, npad, 1.3)]
mesh = Mesh.CylMesh(hx, hy, '00D')

# Define geologic parameters and electrical conductivity
k0 = 1e-12
mu_r = 1e-1
sigma = 1e-2

# Define relative magnetic permeability
mu_0 = 4e-7 * np.pi

# Define mappings
mapping = Maps.ExpMap(mesh)

# Set up the FDEM problem and survey
survey = FDEM.Survey()
survey.add_frequency_data(np.logspace(-2, 2, 20), np.logspace(-2, 2, 20))

# Set up the FDEM problem
prob = FDEM.Problem3D_CC(mesh, sigmaMap=mapping, bc_type='Neumann',
                         solver=SolverPardiso)
prob.pair(survey)

# Perform the FDEM inversion
m0 = np.log(1e-8) * np.ones(mesh.nC)
mref = np.log(1e-8) * np.ones(mesh.nC)

# Set up inversion directives
opt = {}
opt['max_iter'] = 20
opt['tol'] = 1e-6
opt['verbose'] = False
opt['beta_tol'] = 1e-6
opt['beta_init'] = 1e-8
opt['beta_adjust'] = False
opt['cooling_rate'] = 2
opt['cooling_type'] = 'basic'
opt['alpha_s'] = 1.0
opt['alpha_x'] = 1.0
opt['alpha_y'] = 1.0
opt['alpha_z'] = 1.0
opt['gamma_s'] = 0.0
opt['gamma_x'] = 0.0
opt['gamma_y'] = 0.0
opt['gamma_z'] = 0.0
opt['f_min_change'] = 1e-4
opt['x_tol'] = 1e-6
opt['f_tol'] = 1e-6
opt['solver'] = SolverPardiso
opt['max_ls_iter'] = 10
opt['ls_tol'] = 1e-6
opt['ls_step'] = 0.5
opt['ls_max'] = 1.5
opt['ls_min'] = 1e-5
opt['ls_strict'] = False
opt['ls_initial'] = 1.0
opt['ls_verbose'] = False
opt['ls_max_iter'] = 10
opt['ls_max_iter_bkp'] = 10
opt['ls_max_iter_bkp_ratio'] = 2
opt['ls_max_iter_bkp_init'] = 10
opt['ls_max_iter_bkp_init_ratio'] = 2
opt['ls_max_iter_bkp_init_step'] = 0.5
opt['ls_max_iter_bkp_init_step_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min'] = 1e-4
opt['ls_max_iter_bkp_init_step_min_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step'] = 0.5
opt['ls_max_iter_bkp_init_step_min_ratio_step_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min'] = 1e-6
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step'] = 0.5
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min'] = 1e-8
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step'] = 0.5
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min'] = 1e-10
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step'] = 0.5
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min'] = 1e-12
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step'] = 0.5
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min'] = 1e-14
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step'] = 0.5
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_ratio'] = 2
opt['ls_max_iter_bkp_init_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min_ratio_step_min'] = 