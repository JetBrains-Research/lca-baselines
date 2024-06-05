```python
import dd4hep
import dd4hep.simulation as sim
import sys

def display_help():
    print("Usage: python simulation.py [options]")
    print("Options:")
    print("  --help                Display this help message")
    print("  --vis                 Enable visualization")
    print("  --geometry=<file>     Specify the geometry file to use")
    print("  --enable-magnetic-field  Enable magnetic field tracking")
    print("  --random-seed=<value> Set the random seed")

def main():
    # Setup the logger
    dd4hep.setPrintLevel(dd4hep.printout.INFO)
    
    # Parse command line arguments
    cmd = sim.CommandLine()
    cmd.registerArg("help", display_help)
    cmd.registerArg("vis", bool)
    cmd.registerArg("geometry", str)
    cmd.registerArg("enable-magnetic-field", bool)
    cmd.registerArg("random-seed", int)
    args = cmd.parse()

    if args.get("help", False):
        display_help()
        sys.exit(0)

    # Setup the Geant4 kernel and detector description
    kernel = sim.DD4hepSimulation()

    # Configure the user interface
    # This is a placeholder for user interface configuration

    # Load specific geometry
    if "geometry" in args:
        kernel.loadGeometry(args["geometry"])

    # Setup magnetic field tracking if enabled
    if args.get("enable-magnetic-field", False):
        sim.setupTrackingField(kernel)

    # Setup random generator
    sim.setupUserPhysics(kernel, {"random.seed": args.get("random-seed", 12345)})

    # Event actions, I/O, and various generator actions
    # This is a placeholder for event actions, I/O, and generator actions setup

    # Setup detectors, build the physics list, add special particle types and a global range cut
    # This is a placeholder for detectors, physics list, special particles, and range cut setup

    # If visualization is enabled
    if args.get("vis", False):
        # This is a placeholder for visualization commands

    # Configure, initialize, run, and terminate the kernel
    sim._configureDD4HEP(kernel, args)
    sim.commandsInitialize(kernel)
    kernel.initialize()
    kernel.run()
    sim.commandsTerminate(kernel)
    sim._kernel_terminate(kernel)

if __name__ == "__main__":
    main()
```