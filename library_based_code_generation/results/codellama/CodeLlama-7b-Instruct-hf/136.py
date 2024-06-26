  import dream

dream.set_electric_field_strength(0.6)
dream.set_electron_density(5e19)
dream.set_temperature(1e3)
dream.add_ion_species('D', charge=1)
dream.disable_hot_tail_grid()
dream.set_collision_frequency_mode('ultra-relativistic')
dream.include_avalanche(mode='fluid', rate='neural network')
dream.include_dreicer()
dream.set_initial_profile(1e15)
dream.enable_runaway_electron_grid(50, 100, 0.5)
dream.set_advection_interpolation_method('flux limiters')
dream.set_initialization_method('isotropic')
dream.set_radial_grid(5, 0.22, 0.22, 1)
dream.set_solver('nonlinear', 'verbose', relative_tolerance=1e-4)
dream.set_time_stepper(max_time=1e-1, num_steps=20)
dream.save_settings('dream_settings.h5')