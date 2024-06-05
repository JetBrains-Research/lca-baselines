import numpy as np
import dream
from dream.simulation import Simulation

settings = dream.run.default_settings()

settings.physics.E_field = 0.6
settings.physics.n_cold = 5e19
settings.physics.T_cold = 1e3
settings.physics.IonSpecies['D'] = 1
settings.physics.hot_tail_grid_enabled = False
settings.physics.collision_model = 'ultra_relativistic'
settings.physics.avalanche_mode = 'fluid'
settings.physics.dreicer_mode = 'neural_network'
settings.physics.initial_profile = 1e15

settings.physics.runaway_grid.radial_points = 50
settings.physics.runaway_grid.momentum_points = 100
settings.physics.runaway_grid.max_momentum = 0.5

settings.physics.advection_interpolation_method = 'flux_limiters'
settings.physics.initialization_method = 'isotropic'

settings.radial_grid.B0 = 5
settings.radial_grid.a = 0.22
settings.radial_grid.wall_radius = 0.22
settings.radial_grid.nr = 1

settings.solver.type = 'nonlinear'
settings.solver.verbose = True
settings.solver.runaway_current_density_tolerance = 1e-4

settings.physics.fluid = True

settings.time_stepper.max_time = 1e-1
settings.time_stepper.nt = 20

dream.save(settings, 'dream_settings.h5')