```python
import dd4hep
import dd4hep.simulation as sim
import logging

def run_simulation():
    from dd4hep import CommandLine
    import sys
    import dd4hep.simulation as sim
    import dd4hep.kernel as kernel

    # Set up command line arguments
    cl = CommandLine()
    cl.parse()

    if cl.has_arg("help"):
        cl.print_help()
        sys.exit(0)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load geometry
    geometry_file = cl.opt("geometry", "", "Path to the geometry file")
    if not geometry_file:
        logging.error("Geometry file is required.")
        sys.exit(1)

    # Load constants and set up Geant4
    kernel.load_DD4hep_geometry(geometry_file)
    sim.importConstants()

    # Print detectors
    sim.printDetectors()

    # Configure UI
    sim.setPrintLevel(sim.INFO)
    sim.setOption("DD4hepUI", True)

    # Tracking field and event actions
    sim.__setMagneticFieldOptions()
    sim.enableDetailedHitsAndParticleInfo()

    # Particle gun and tracker
    gun = sim.DD4hepSimulation.configureParticleGun(cl)
    tracker = sim.DD4hepSimulation.setupTracker()

    # Build physics list and execute Geant4
    physics_list = sim.DD4hepSimulation.setupPhysicsList(cl.opt("physicsList", "FTFP_BERT", "Name of the physics list"))
    sim.DD4hepSimulation.execute()

if __name__ == "__main__":
    run_simulation()
```