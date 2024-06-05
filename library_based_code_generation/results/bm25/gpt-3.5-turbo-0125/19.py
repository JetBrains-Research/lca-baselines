import dream
import numpy as np

eq = dream.Equilibrium()
eq.setB0(5)  # Set magnetic field strength to 5 T

eq.setN(5e19)  # Set electron density to 5e19 m^-3
eq.setT(100)  # Set temperature to 100 eV

eq.setElectricField(6)  # Set electric field strength to 6 V/m

eq.setRadialGrid(dream.RadialGrid(0, 5, 100))  # Set up radial grid

eq.setSolver(dream.Solver.LINEAR_IMPLICIT)  # Set solver type

eq.setTimeStepper(dream.TimeStepper.EXPLICIT)  # Set time stepper

eq.setSaveSteps(20)  # Set number of save steps

eq.setRunawayGrid(dream.Grid(0, 1, 300, 20))  # Set up momentum and pitch grid

eq.setSimulationTime(1e-3)  # Set simulation time to 1e-3 seconds

eq.setOutputFileName('output.h5')  # Set output file name

eq.run()  # Run simulation