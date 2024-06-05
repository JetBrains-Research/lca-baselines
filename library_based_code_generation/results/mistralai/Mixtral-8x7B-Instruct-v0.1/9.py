 ```python
import argparse
import logging
import dd4hep
from dd4hep import DD4hepDetector
from dd4hep import simulate
from dd4hep import PhysicalVolume
from dd4hep import segmentation as seg
from dd4hep import schütz as sch

def run_simulation(args):
    if args.help:
        print("Help message")
        return

    # Load geometry
    det_factory = DD4hepDetector('my_detector', args.geom_file)

    # Load constants
    dd4hep.setLatticeConstant('m', 1.0)

    # Initialize Geant4
    geant4 = simulate.Geant4(det_factory, world_type='sensitive', log_level=logging.INFO)

    # Print detectors
    geant4.printDetectors()

    # Configure UI
    geant4.configureUI(visible=False, select=False)

    # Configure tracking field
    geant4.configureTrackingField(type='constantMagneticField', strength=[0., 0., 0.1])

    # Configure event actions
    geant4.configureEventAction(simulate.FastSimulationAction())

    # Set up particle gun
    gun = geant4.particleGun()
    gun.particleType(geant4.Geant4ParticleType.GEANT4_PIMINUS)
    gun.momentum([0., 0., 10.])
    gun.position([0., 0., 0.5])

    # Set up tracker
    tracker = PhysicalVolume(det_factory.world(), name='Tracker')
    tracker.segmentation(seg.SimpleRectangular(dx=1.0, dy=1.0, dz=1.0))
    tracker.sensitiveDetector(sch.FastSimulationSD(tracker))

    # Build physics list
    geant4.buildPhysicsList()

    # Execute Geant4
    geant4.simulateEvent(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulation using dd4hep.')
    parser.add_argument('--geom-file', type=str, help='Path to the geometry file.')
    parser.add_argument('--help', action='store_true', help='Print help message.')
    args = parser.parse_args()
    run_simulation(args)
```