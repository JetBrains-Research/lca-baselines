  import os
import sys
from dd4hep.simulations import DDG4
from dd4hep.detectors import Detector
from dd4hep.geometry import Geometry
from dd4hep.units import units
from dd4hep.simulations.g4 import G4Simulation
from dd4hep.simulations.g4.g4_detector import G4Detector
from dd4hep.simulations.g4.g4_particle_gun import G4ParticleGun
from dd4hep.simulations.g4.g4_tracking_field import G4TrackingField
from dd4hep.simulations.g4.g4_event_actions import G4EventActions
from dd4hep.simulations.g4.g4_detector_construction import G4DetectorConstruction
from dd4hep.simulations.g4.g4_physics_list import G4PhysicsList
from dd4hep.simulations.g4.g4_ui import G4UI

def setup_dd4hep_simulation():
    # Initialize DDG4 kernel
    dd4hep_kernel = DDG4()

    # Load geometry from XML file
    geometry = Geometry(dd4hep_kernel)
    geometry.load_from_file(os.path.join(os.environ['DD4hepExamplesINSTALL'], 'OpticalSurfaces', 'compact', 'compact.xml'))

    # Import constants from detector description
    constants = dd4hep_kernel.detector_description.constants

    # Configure Geant4 instance
    g4_simulation = G4Simulation(dd4hep_kernel)
    g4_simulation.tracking_field = G4TrackingField(constants.tracking_field_length)
    g4_simulation.event_actions = G4EventActions()
    g4_simulation.detector_construction = G4DetectorConstruction(constants.detector_material)
    g4_simulation.detector_construction.add_detector(G4Detector(constants.detector_name, constants.detector_material))
    g4_simulation.particle_gun = G4ParticleGun(constants.particle_gun_energy, constants.particle_gun_multiplicity)
    g4_simulation.particle_gun.add_particle(constants.particle_gun_particle)

    # Set up UI
    if len(sys.argv) > 1:
        g4_simulation.ui = G4UI(sys.argv[1])

    # Set up tracker and physics list
    g4_simulation.detector_construction.add_tracker(G4Tracker('MaterialTester', constants.tracker_material))
    g4_simulation.physics_list = G4PhysicsList('QGSP_BERT')

    # Execute Geant4 instance
    g4_simulation.execute()

if __name__ == '__main__':
    setup_dd4hep_simulation()