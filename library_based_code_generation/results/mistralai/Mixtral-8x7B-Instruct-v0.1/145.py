 import dream

dream.set_fluid_options(hot_tail_grid=True, runaway_grid=True)
dream.set_kinetic_options(efield_strength=5.0, electron_density=1e19, temperature=10.0)
dream.set_momentum_grid(num_points=256, min_momentum=0.0, max_momentum=20.0)
dream.set_maxwellian(num_ species=1, num_energy_bins=256, temperature=10.0)
dream.set_reaction_options(include_dreicer=True, include_avalanche=True)
dream.set_radial_grid(num_points=100, lower_boundary=0.0, upper_boundary=1.0)
dream.set_runaway_options(use_grid=False)
dream.set_transport_options(use_svensson=True)
dream.set_solver_options(nonlinear_solver='picard')
dream.set_timestepper_options(timestep_method='rk4', timestep_size=1e-8)
dream.save_settings('simulation.h5')