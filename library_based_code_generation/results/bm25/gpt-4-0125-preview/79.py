```python
import os
import sys
from DDG4 import Kernel, Geant4
from DDG4 import ParticleGun, PhysicsList, UI
from DDG4 import Geometry

def initialize_dd4hep_simulation():
    kernel = Kernel()
    install_path = os.environ['DD4hepExamplesINSTALL']
    geometry_file = os.path.join(install_path, 'OpticalSurfaces', 'compact', 'geometry.xml')
    kernel.loadGeometry(geometry_file)

    # Import constants from the kernel's detector description
    from DD4hep import Simulation
    constants = Simulation.constants(kernel.detectorDescription())

    # Configure Geant4 instance
    g4 = Geant4(kernel)
    g4.trackingField.magneticField = 'ConstantMagneticField'
    g4.addDetectorConstruction(Geometry(kernel))
    g4.addPhysicsList(PhysicsList(kernel, 'QGSP_BERT'))
    g4.addEventAction(ParticleGun(kernel, particle='gamma', energy=5e3, multiplicity=1))
    g4.setupTracker('MaterialTester')
    
    # UI setup
    if len(sys.argv) > 1:
        ui = UI(kernel)
        ui.sessionType = 'VIS'
        ui.setupWithMacro(sys.argv[1])
    else:
        g4.setupUI()

    # Execute the Geant4 instance
    g4.execute()

if __name__ == '__main__':
    initialize_dd4hep_simulation()
```