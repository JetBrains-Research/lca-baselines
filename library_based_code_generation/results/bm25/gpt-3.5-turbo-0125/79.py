```python
import os
from dd4hep import DDG4

def setup_dd4hep_simulation():
    kernel = DDG4.Kernel()
    kernel.loadGeometry(os.path.join(os.environ['DD4hepExamplesINSTALL'], 'OpticalSurfaces/compact/MyDetector.xml'))
    kernel.detectorDescription()

    geant4 = DDG4.Geant4(kernel, tracker='MaterialTester', physics='QGSP_BERT')
    geant4.setupTrackingField()
    geant4.setupEventAction()
    geant4.setupDetectorConstruction()
    geant4.setupParticleGun(particle='gamma', energy=5, multiplicity=1)
    geant4.setupUI(macro=None)

    geant4.run()

if __name__ == "__main__":
    setup_dd4hep_simulation()
```