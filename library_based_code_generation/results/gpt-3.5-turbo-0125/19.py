import numpy as np
import dream
eq = dream.EquationSystem()
eq.setInitialProfiles(n=5e19, T=100, V=6)
eq.setBoundaryCondition('radial', 'mirror')
eq.setRadialGrid(0, 1, 300)
eq.setPitchGrid(20)
eq.setSolverType('implicit')
eq.setTimeStepper(timestep=1e-5, steps=20)
eq.setSaveOutput('output.h5')
eq.run()