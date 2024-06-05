  Here is the Python code that sets up a dd4hep simulation using the provided instructions:
```
import os
from dd4hep.simulations import DDG4
from dd4hep.simulations.detector import DetectorConstruction
from dd4hep.simulations.particle import ParticleHandler
from dd4hep.simulations.geometry import Geometry
from dd4hep.simulations.tracking import TrackingField
from dd4hep.simulations.event import EventActions
from dd4hep.simulations.particle_gun import Gun
from dd4hep.simulations.physics_list import PhysicsList
from dd4hep.simulations.ui import UI

def setup_dd4hep_simulation():
    # Initialize DDG4 kernel
    kernel = DDG4()

    # Load geometry from XML file
    geometry = Geometry.from_file(os.path.join(os.environ['DD4hepExamplesINSTALL'], 'OpticalSurfaces', 'compact', 'geometry.xml'))

    # Import constants from detector description
    kernel.import_constants(geometry.detector_description)

    # Configure Geant4 instance
    geant4 = Geant4()
    geant4.set_tracking_field(TrackingField.from_geometry(geometry))
    geant4.set_event_actions(EventActions.from_geometry(geometry))
    geant4.set_detector_construction(DetectorConstruction.from_geometry(geometry))
    geant4.set_particle_gun(Gun.from_geometry(geometry))

    # Set up UI
    if len(sys.argv) > 1:
        ui = UI.from_macro(sys.argv[1])
    else:
        ui = UI()

    # Set up particle gun
    particle_gun = ParticleHandler.from_geometry(geometry)
    particle_gun.set_particle(Particle.from_name('gamma'))
    particle_gun.set_energy(5.0 * keV)
    particle_gun.set_multiplicity(1)

    # Set up tracker and physics list
    tracker = DetectorConstruction.from_geometry(geometry)
    physics_list = PhysicsList.from_geometry(geometry)

    # Execute Geant4 instance
    geant4.execute(ui, tracker, physics_list)

if __name__ == '__main__':
    setup_dd4hep_simulation()
```
This code defines a function `setup_dd4hep_simulation` that sets up a dd4hep simulation using the provided instructions. The function initializes a DDG4 kernel, loads a geometry from an XML file located in the 'OpticalSurfaces/compact' directory of the 'DD4hepExamplesINSTALL' environment variable, and imports constants from the kernel's detector description. It then configures a Geant4 instance with a tracking field, event actions, detector construction, and a particle gun. The Geant4 instance is set up with a UI, which uses a macro if provided as a command line argument. The particle gun is set up with a gamma particle, an energy of 5 keV, and a multiplicity of 1. The function also sets up a tracker named 'MaterialTester' and a physics list named 'QGSP_BERT'. After all configurations, the function executes the Geant4 instance. Finally, the code calls the `setup_dd4hep_simulation` function if it is the main module being run.