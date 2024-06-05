 import ddg4
import dd4hep as dd
import sys

def run_simulation():
    # Initialize DDG4 kernel
    kernel = ddg4.DDG4(geant4_version="10.7.p02", nthreads=4)

    # Load geometry from XML file
    geo_file = f"{dd.env.DD4hepExamplesINSTALL}/OpticalSurfaces/compact/opticalsurfaces.xml"
    det_desc = kernel.detdesc()
    det_desc.setGeometry(dd.read(geo_file))

    # Import constants from detector description
    det_desc.importConstants()

    # Configure Geant4 instance
    g4inst = kernel.g4inst()

    # Set up tracking field
    tracking_field = ddg4.FieldFactory.createTrackingField(det_desc)
    g4inst.setTrackingField(tracking_field)

    # Set up event actions
    event_action = ddg4.MyEventAction()
    g4inst.setUserAction(event_action)

    # Set up detector construction
    detector = ddg4.MyDetectorConstruction()
    g4inst.setDetectorConstruction(detector)

    # Set up particle gun
    gun = ddg4.G4ParticleGun(kernel)
    gun.setParticleDefinition("gamma")
    gun.setParticleEnergy(5*dd.keV)
    gun.setParticleMultiplicity(1)
    g4inst.setGun(gun)

    # Set up tracker
    tester = ddg4.MyTracker("MaterialTester")
    g4inst.addTracker(tester)

    # Set up physics list
    physics_list = "QGSP_BERT"
    g4inst.setPhysicsList(physics_list)

    # Execute Geant4 instance
    g4inst.execute()

if __name__ == "__main__":
    run_simulation()