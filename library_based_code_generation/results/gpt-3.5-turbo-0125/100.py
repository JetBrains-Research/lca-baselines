```python
from dd4hep import DDG4

kernel = DDG4.Kernel()
kernel.loadGeometry("file:$DD4hepExamplesINSTALL/Path/To/Your/Geometry.xml")

lcdd = kernel.detectorDescription()

geant4 = DDG4.Geant4(kernel, tracker='Geant4TrackerCombineAction')
geant4.setupCshUI()

geant4.trackingField = 'Geant4FieldTracking'
geant4.eventAction = 'Geant4EventAction'
geant4.generatorAction = 'Geant4ParticleGun'

gun = geant4.setupGun("Gun", particle='gamma', energy=5, multiplicity=1)

tracker = geant4.setupTracker('Geant4Tracker')
physics = geant4.setupPhysics('Geant4PhysicsList')
physics.particle = ['gamma', 'e-', 'e+']
physics.enableUI()

kernel.configure()
kernel.initialize()
kernel.run()
kernel.terminate()
```