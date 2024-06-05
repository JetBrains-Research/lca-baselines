  import logging
import argparse
import dd4hep

logging.basicConfig(level=logging.INFO)

def run_simulation():
    # Import additional modules
    import Geant4
    import G4RunManager
    import G4UImanager
    import G4EventManager
    import G4TrackingManager
    import G4ParticleGun
    import G4Tracker
    import G4PhysicsList

    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--geometry", help="Path to geometry file")
    parser.add_argument("-c", "--constants", help="Path to constants file")
    parser.add_argument("-p", "--particle", help="Particle to simulate")
    parser.add_argument("-n", "--num-events", help="Number of events to simulate")
    parser.add_argument("-o", "--output", help="Path to output file")
    args = parser.parse_args()

    # Load geometry file
    if args.geometry:
        geometry = dd4hep.load(args.geometry)
    else:
        logging.error("No geometry file specified")
        return

    # Load constants file
    if args.constants:
        constants = dd4hep.load(args.constants)
    else:
        logging.error("No constants file specified")
        return

    # Set up Geant4
    G4RunManager.initialize()
    G4UImanager.initialize()
    G4EventManager.initialize()
    G4TrackingManager.initialize()

    # Set up particle gun
    particle_gun = G4ParticleGun()
    particle_gun.SetParticle(args.particle)

    # Set up tracker
    tracker = G4Tracker()

    # Set up physics list
    physics_list = G4PhysicsList()
    physics_list.SetPhysicsParameters(constants)

    # Configure UI
    G4UImanager.getUIpointer()->ApplyCommand("/control/execute/runAction", "1");
    G4UImanager.getUIpointer()->ApplyCommand("/control/execute/eventAction", "1");

    # Configure tracking field
    G4TrackingManager.getTrackingManager()->SetTrackingConfiguration(constants);

    # Configure event actions
    G4EventManager.getEventManager()->SetEventAction(G4TrackingManager.getTrackingManager());

    # Build physics list
    physics_list.BuildPhysicsList();

    # Execute Geant4
    G4RunManager.getRunManager()->SetUserInitialization(physics_list);
    G4RunManager.getRunManager()->SetUserAction(particle_gun);
    G4RunManager.getRunManager()->SetUserAction(tracker);
    G4RunManager.getRunManager()->BeamOn(args.num_events);

if __name__ == "__main__":
    run_simulation()