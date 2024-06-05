 ```python
import logging
import dd4hep
import dd4hep.detector as detector
import dd4hep.spec as spec
import dd4hep.geometry_readout as geometry_readout
import dd4hep.segmentation as segmentation
import dd4hep.core.Constants as Constants
import Geant4
import Geant4.hepUnits as hepUnits

def run_simulation():
    # Import additional modules
    from Configurables import HepMCGeninterface, Simulation, Geant4Kernel

    # Set up the kernel
    kernel = Geant4Kernel("Geant4Kernel")

    # Load the geometry from a file
    det_descr = detector.Detector("MyDetector", **spec.ideal_hcal_digital.geometry)
    det_descr.description = "My custom detector description"
    readout = geometry_readout.GeometryReadout("MyReadout")
    readout.description = "My custom readout"
    readout.addVolume(det_descr)
    kernel.geometry = readout

    # Import constants
    kernel.constants = Constants

    # Configure the Geant4 interface
    g4_interface = kernel.g4interface
    g4_interface.DetectorVersion = "FTFP_BERT"
    g4_interface.MaxStep = 0.1 * hepUnits.mm
    g4_interface.ProduceNtuples = True
    g4_interface.NtupleFile = "simulation.root"

    # Set up the tracking field
    tracking_field = kernel.world.field
    tracking_field.provider = "SimpleMagneticField"
    tracking_field.SimpleMagneticField.Bfield = 0.1 * hepUnits.T

    # Set up event actions
    event_action = kernel.world.event_action
    event_action.add(Geant4.MyEventAction())

    # Set up the particle gun
    particle_gun = kernel.world.gun
    particle_gun.ParticleGun.ParticleType = "geant4. g4Proton"
    particle_gun.ParticleGun.Energy = 100.0 * hepUnits.GeV
    particle_gun.ParticleGun.Position = (0.0, 0.0, 0.0) * hepUnits.cm
    particle_gun.ParticleGun.Direction = (0.0, 0.0, 1.0)

    # Handle simulation particles
    simulation_action = kernel.world.action
    simulation_action.add(Geant4.MySimulationAction())

    # Build the physics list
    physics_list = kernel.world.physics
    physics_list.add(Geant4.MyPhysicsList())

    # Start the engine
    engine = Simulation()
    engine.world = kernel.world
    engine.components = [HepMCGeninterface]
    engine.services = [Geant4Kernel]
    engine.geant4Kernel.geant4 = kernel

    return engine

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = run_simulation()
    engine.run(100)
```