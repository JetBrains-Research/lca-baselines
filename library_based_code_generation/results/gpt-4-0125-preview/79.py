```python
import os
import sys
import DDG4
from DDG4 import Geometry

def setup_dd4hep_simulation():
    # Initialize the DDG4 kernel
    kernel = DDG4.Kernel()
    description = kernel.detectorDescription()

    # Load geometry
    geometry_file = os.path.join(os.environ['DD4hepExamplesINSTALL'], 'OpticalSurfaces', 'compact', 'geometry.xml')
    Geometry.configureFromXML(geometry_file, kernel)

    # Import constants from the detector description
    DDG4.importConstants(description)

    # Configure Geant4
    geant4 = DDG4.Geant4(kernel)
    geant4.load('libG4processes')
    geant4.trackingField = True
    geant4.addDetectorConstruction(DDG4.DetectorConstruction(kernel, description))
    geant4.addPhysicsList('QGSP_BERT')
    geant4.addEventAction(DDG4.EventAction(kernel, 'Geant4EventAction'))
    
    # Particle gun
    particle_gun = DDG4.ParticleGun(kernel, 'ParticleGun')
    particle_gun.particle = 'gamma'
    particle_gun.energy = 5 * DDG4.keV
    particle_gun.multiplicity = 1
    geant4.addPrimaryGenerator(particle_gun)

    # Tracker
    tracker = DDG4.Action(kernel, 'MaterialTester', 'Tracker')
    kernel.registerGlobalAction(tracker)

    # UI
    if len(sys.argv) > 1:
        geant4.ui = sys.argv[1]

    # Execute
    geant4.execute()

if __name__ == '__main__':
    setup_dd4hep_simulation()
```