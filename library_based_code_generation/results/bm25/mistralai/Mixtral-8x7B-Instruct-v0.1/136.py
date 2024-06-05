 import dream

dream\_settings = dream.Settings()

dream\_settings.electricFieldStrength = 0.6
dream\_settings.electronDensity = 5e19
dream\_settings.temperature = 1e3

dream\_settings.ionNames = ['D']
dream\_settings.ionCharges = [1]

dream\_settings.hotTailGrid = False
dream\_settings.collisionFrequencyMode = dream.CollisionFrequencyMode.ULTRA\_RELATIVISTIC

dream\_settings.runawayElectronGrid.enabled = True
dream\_settings.runawayElectronGrid.numberOfRadialPoints = 50
dream\_settings.runawayElectronGrid.numberOfMomentumPoints = 100
dream\_settings.runawayElectronGrid.maximumMomentum = 0.5

dream\_settings.advectionInterpolationMethod = dream.AdvectionInterpolationMethod.FLUX\_LIMITER
dream\_settings.initializationMethod = dream.InitializationMethod.ISOTROPIC

dream\_settings.radialGrid = dream.RadialGrid(
    magneticFieldStrength=5,
    minorRadius=0.22,
    wallRadius=0.22,
    numberOfRadialPoints=1
)

dream\_settings.runawayElectronDensity = dream.RunawayElectronDensity(
    include=dream.RunawayElectronDensityComponents.DRIECER | dream.RunawayElectronDensityComponents.AVALANCHE,
    dreicer=dream.Dreicer(rateEstimator=dream.DreicerRateEstimators.NEURAL\_NETWORK),
    avalanche=dream.Avalanche(mode=dream.AvalancheModes.FLUID)
)

dream\_settings.runawayElectronDensity.initialProfiles = {
    dream.ParticleTypes.ELECTRON: 1e15
}

dream\_settings.solver = dream.Solver(
    method=dream.SolverMethods.NONLINEAR,
    verbose=True,
    relTol=1e-4
)

dream\_settings.timeStepper = dream.TimeStepper(
    maxTime=1e-1,
    numberOfTimeSteps=20
)

dream\_settings.saveToFile('dream_settings.h5')