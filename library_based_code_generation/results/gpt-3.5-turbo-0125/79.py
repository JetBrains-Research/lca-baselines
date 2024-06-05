```python
import os
from DDG4 import Kernel
from DD4hep import GeoLoader

def setup_dd4hep_simulation(macro=None):
    kernel = Kernel()
    kernel.loadGeometry(os.path.join(os.environ['DD4hepExamplesINSTALL'], 'OpticalSurfaces/compact/MyDetector.xml'))
    dd4hep = kernel.detectorDescription()

    geant4 = kernel.geant4()
    geant4.setupTrackingField()
    geant4.setupEventAction()
    geant4.setupDetectorConstruction()
    geant4.setupParticleGun(particle='gamma', energy=5, multiplicity=1)
    geant4.setupUI(macro=macro)

    tracker = geant4.setupTracker(name='MaterialTester')
    physics_list = geant4.setupPhysics('QGSP_BERT')

    kernel.run()

if __name__ == '__main__':
    setup_dd4hep_simulation()
```